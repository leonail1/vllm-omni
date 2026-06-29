# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for step-level diffusion execution across runner / worker / executor / engine."""

import contextlib
import inspect
import os
import queue
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
import vllm_omni.diffusion.diffusion_engine as diffusion_engine_module
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.comm import RingComm, SeqAllToAll4D
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.ipc import (
    pack_diffusion_output_shm,
    unpack_diffusion_output_shm,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import RequestScheduler, StepScheduler
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionSchedulerOutput,
    NewRequestData,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, DiffusionRequestState, RunnerOutput
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


@contextmanager
def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    yield


def _update_environment_variables(envs_dict: dict[str, str]) -> None:
    for key, value in envs_dict.items():
        os.environ[key] = value


class _StepPipeline:
    """Minimal pipeline stub that supports step-wise execution."""

    supports_step_execution = True

    def __init__(self):
        self.prepare_calls = 0
        self.denoise_calls = 0
        self.scheduler_calls = 0
        self.decode_calls = 0

    def encode_stage(self, state):
        self.prepare_calls += 1
        state.timesteps = [torch.tensor(10), torch.tensor(5)]
        state.latents = torch.tensor([0.0])
        state.prompt_embeds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        return state

    def _state_from_request(self, req):
        return DiffusionRequestState(
            request_id=req.request_id,
            sampling=req.sampling_params,
            prompts=req.prompts,
        )

    def denoise_stage(self, input_batch):
        self.denoise_calls += 1
        return torch.full_like(input_batch.prompt_embeds, fill_value=0.5)

    def scheduler_stage(self, state, noise_pred):
        del noise_pred
        self.scheduler_calls += 1
        state.step_index += 1

    def decode_stage(self, state):
        self.decode_calls += 1
        return DiffusionOutput(output=torch.tensor([state.step_index], dtype=torch.float32))


class _InterruptingStepPipeline(_StepPipeline):
    interrupt = True

    def denoise_stage(self, input_batch):
        del input_batch
        self.denoise_calls += 1
        return None

    def scheduler_stage(self, state, noise_pred):
        del state, noise_pred
        raise AssertionError("scheduler_stage should not run after interrupt")

    def decode_stage(self, state):
        del state
        raise AssertionError("decode_stage should not run after interrupt")


class _IdentityNoiseTransformer(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return (x,)


class _AdditiveScheduler:
    def step(self, noise_pred: torch.Tensor, t: torch.Tensor, latents: torch.Tensor, return_dict: bool = False):
        del t, return_dict
        return (latents + noise_pred,)


class _DistributedStepPipeline(CFGParallelMixin):
    supports_step_execution = True

    def __init__(self, mode: str, device: torch.device):
        self.mode = mode
        self.device = device
        self._interrupt = False
        self.scheduler = _AdditiveScheduler()
        self.transformer = _IdentityNoiseTransformer()

    @property
    def interrupt(self):
        return self._interrupt

    def encode_stage(self, state):
        state.timesteps = [torch.tensor(1.0, device=self.device)]
        state.latents = torch.ones((1, 1), device=self.device)
        state.step_index = 0
        state.scheduler = self.scheduler
        state.do_true_cfg = self.mode == "cfg"
        state.prompt_embeds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        return state

    def denoise_stage(self, input_batch):
        if self.mode == "ulysses":
            sp_group = get_sp_group().ulysses_group
            seq_world_size = torch.distributed.get_world_size(sp_group)
            input_tensor = torch.randn(1, 2, 2 * seq_world_size, 2, device=self.device)
            original = input_tensor.clone()
            intermediate = SeqAllToAll4D.apply(sp_group, input_tensor, 2, 1, False)
            output = SeqAllToAll4D.apply(sp_group, intermediate, 1, 2, False)
            torch.testing.assert_close(output, original, rtol=1e-5, atol=1e-5)
            return torch.ones_like(input_batch.latents)

        if self.mode == "ring":
            ring_group = get_sp_group().ring_group
            rank = torch.distributed.get_rank(ring_group)
            world_size = torch.distributed.get_world_size(ring_group)
            comm = RingComm(ring_group)
            input_tensor = torch.full((1, 2, 2), float(rank + 1), device=self.device)
            recv_tensor = comm.send_recv(input_tensor)
            comm.commit()
            comm.wait()
            expected = torch.full_like(recv_tensor, float(((rank - 1) % world_size) + 1))
            torch.testing.assert_close(recv_tensor, expected, rtol=1e-5, atol=1e-5)
            return torch.ones_like(input_batch.latents)

        positive_kwargs = {"x": input_batch.latents + 1}
        negative_kwargs = {"x": input_batch.latents - 1}
        return self.predict_noise_maybe_with_cfg(
            do_true_cfg=True,
            true_cfg_scale=1.0,
            positive_kwargs=positive_kwargs,
            negative_kwargs=negative_kwargs,
            cfg_normalize=False,
        )

    def scheduler_stage(self, state, noise_pred):
        if self.mode == "cfg":
            state.latents = self.scheduler_step_maybe_with_cfg(
                noise_pred,
                state.current_timestep,
                state.latents,
                do_true_cfg=True,
                per_request_scheduler=state.scheduler,
            )
        else:
            state.latents = state.latents + noise_pred
        state.step_index += 1

    def decode_stage(self, state):
        return DiffusionOutput(output=state.latents.detach().cpu())


def _make_step_request(num_inference_steps: int = 2):
    return SimpleNamespace(
        prompts=["a prompt"],
        request_id="req-1",
        sampling_params=SimpleNamespace(
            generator=None,
            seed=None,
            generator_device=None,
            num_inference_steps=num_inference_steps,
        ),
    )


def _assert_aborted_output(output: DiffusionOutput, request_id: str) -> None:
    assert output.output is None
    assert output.error is None
    assert output.aborted is True
    assert output.abort_message == f"Request {request_id} aborted."


def _make_engine_request(req_id: str = "req-1", num_inference_steps: int = 2) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[f"prompt-{req_id}"],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=num_inference_steps),
        request_id=req_id,
    )


def _make_vllm_config():
    @contextlib.contextmanager
    def set_priority(*args, **kwargs):
        yield

    return SimpleNamespace(
        kernel_config=SimpleNamespace(ir_op_priority=SimpleNamespace(set_priority=set_priority)),
        compilation_config=SimpleNamespace(ir_enable_torch_wrap=True),
    )


def _make_v2_runner(
    *,
    step_execution: bool = True,
    stage_split: bool = False,
    stage_role: str = "all",
    pipeline=None,
    device: torch.device | None = None,
):
    from vllm_omni.diffusion.worker.diffusion_model_runner_v2 import DiffusionModelRunnerV2

    runner = object.__new__(DiffusionModelRunnerV2)
    runner.vllm_config = _make_vllm_config()
    runner.od_config = SimpleNamespace(
        cache_backend=None,
        parallel_config=SimpleNamespace(use_hsdp=False),
        step_execution=step_execution,
        stage_split=stage_split,
        stage_role=stage_role,
        streaming_output=False,
    )
    runner.device = torch.device("cpu") if device is None else device
    runner.pipeline = _StepPipeline() if pipeline is None else pipeline
    runner.cache_backend = None
    runner.offload_backend = None
    runner.state_cache = {}
    runner.kv_transfer_manager = SimpleNamespace()
    return runner


def _make_scheduler_output(req, request_id="req-1", step_id=0, finished_req_ids=None):
    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=[NewRequestData(request_id=request_id, req=req)],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set() if finished_req_ids is None else set(finished_req_ids),
        num_running_reqs=1,
        num_waiting_reqs=0,
    )


def _make_batch_scheduler_output(reqs, *, step_id=0, finished_req_ids=None):
    """Scheduler output for a homogeneous batch (one NewRequestData per req)."""
    new_reqs = [NewRequestData(request_id=r.request_id, req=r) for r in reqs]
    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set() if finished_req_ids is None else set(finished_req_ids),
        num_running_reqs=len(new_reqs),
        num_waiting_reqs=0,
    )


def _make_cached_scheduler_output(request_id="req-1", step_id=1, finished_req_ids=None):
    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(request_ids=[request_id]),
        finished_req_ids=set() if finished_req_ids is None else set(finished_req_ids),
        num_running_reqs=1,
        num_waiting_reqs=0,
    )


def _make_engine(scheduler, execute_fn=None) -> DiffusionEngine:
    engine = object.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(model_class_name="QwenImagePipeline")
    engine.pre_process_func = None
    engine.post_process_func = None
    engine.scheduler = scheduler
    engine.execute_fn = execute_fn
    engine._rpc_lock = threading.RLock()
    engine._cv = threading.Condition(engine._rpc_lock)
    engine._closed = False
    engine.abort_queue = queue.Queue()
    return engine


def _expected_output_for_mode(mode: str) -> torch.Tensor:
    if mode == "cfg":
        return torch.tensor([[3.0]])
    return torch.tensor([[2.0]])


def _distributed_step_worker(local_rank: int, world_size: int, mode: str, master_port: str):
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    _update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": master_port,
        }
    )
    model_runner_module.set_forward_context = _noop_forward_context

    try:
        init_distributed_environment()
        if mode == "ulysses":
            initialize_model_parallel(ulysses_degree=world_size)
        elif mode == "ring":
            initialize_model_parallel(ring_degree=world_size)
        elif mode == "cfg":
            initialize_model_parallel(cfg_parallel_size=world_size)
        else:
            raise ValueError(f"Unsupported distributed test mode: {mode}")

        from vllm_omni.diffusion.worker import diffusion_model_runner_v2 as model_runner_v2_module

        model_runner_v2_module.set_forward_context = _noop_forward_context
        encode_pipeline = _DistributedStepPipeline(mode=mode, device=device)
        state = DiffusionRequestState(
            request_id="req-1",
            sampling=_make_step_request(num_inference_steps=1).sampling_params,
            prompts=["a prompt"],
        )
        encode_pipeline.encode_stage(state)
        encode_payload = state.to_transport("encode_to_dit")

        runner = _make_v2_runner(
            step_execution=True,
            stage_split=True,
            stage_role="dit",
            pipeline=_DistributedStepPipeline(mode=mode, device=device),
            device=device,
        )
        transport_request = _make_step_request(num_inference_steps=1)
        transport_request.stage_transport_payload = encode_payload
        result = runner.execute_stepwise(_make_scheduler_output(transport_request, step_id=0))
        output = result.get_request_output("req-1")

        assert output.finished is True
        assert output.result is not None
        decode_runner = _make_v2_runner(
            step_execution=True,
            stage_split=True,
            stage_role="decode",
            pipeline=_DistributedStepPipeline(mode=mode, device=device),
            device=device,
        )
        final_output = decode_runner.execute_decode(output.result.custom_output["stage_transport"])
        torch.testing.assert_close(final_output.output, _expected_output_for_mode(mode), rtol=1e-5, atol=1e-5)
        assert "req-1" not in runner.state_cache
    finally:
        destroy_distributed_env()


# ---------------------------------------------------------------------------
# Runner / Worker
# ---------------------------------------------------------------------------


@pytest.mark.cpu
class TestRunner:
    """Legacy DiffusionModelRunner step-only path is no longer valid."""

    def test_standalone_stepwise_method_is_removed(self):
        assert not hasattr(DiffusionModelRunner, "execute_stepwise")


@pytest.mark.cpu
class TestRunnerV2:
    """DiffusionModelRunnerV2 stage-split routing."""

    def test_state_transport_moves_sampling_tensors_out_of_sampling_metadata(self):
        sampling = OmniDiffusionSamplingParams()
        sampling.latents = torch.tensor([[1.0]])
        sampling.image_latent = torch.tensor([[2.0]])
        sampling.trajectory_timesteps = [torch.tensor([3.0])]
        sampling.output = torch.tensor([[4.0]])
        state = DiffusionRequestState(request_id="req-1", sampling=sampling, prompts=["prompt"])
        state.latents = torch.tensor([[5.0]])
        state.timesteps = torch.tensor([6.0])

        payload = state.to_transport("encode_to_dit")

        assert payload.sampling.generator is None
        assert payload.sampling.latents is None
        assert payload.sampling.image_latent is None
        assert payload.sampling.trajectory_timesteps is None
        assert payload.sampling.output is None
        assert torch.equal(payload.tensors["sampling.latents"], torch.tensor([[1.0]]))
        assert torch.equal(payload.tensors["sampling.image_latent"], torch.tensor([[2.0]]))
        assert torch.equal(payload.tensors["sampling.trajectory_timesteps"][0], torch.tensor([3.0]))
        assert torch.equal(payload.tensors["sampling.output"], torch.tensor([[4.0]]))

        restored = DiffusionRequestState.from_transport(payload, device=torch.device("cpu"))

        assert torch.equal(restored.sampling.latents, torch.tensor([[1.0]]))
        assert torch.equal(restored.sampling.image_latent, torch.tensor([[2.0]]))
        assert torch.equal(restored.sampling.trajectory_timesteps[0], torch.tensor([3.0]))
        assert torch.equal(restored.sampling.output, torch.tensor([[4.0]]))

    def test_stage_pool_requires_stage_transport_key(self):
        payload = {"state": "encoded"}

        assert StagePool._extract_stage_transport_payload({"stage_transport": payload}) is payload
        with pytest.raises(ValueError, match="stage_transport"):
            StagePool._extract_stage_transport_payload({"payload": payload})

    def test_stage_step_runs_encode_dit_decode_transport(self, monkeypatch):
        encode_runner = _make_v2_runner(step_execution=True, stage_split=True, stage_role="encode")
        dit_runner = _make_v2_runner(step_execution=True, stage_split=True, stage_role="dit")
        decode_runner = _make_v2_runner(step_execution=True, stage_split=True, stage_role="decode")
        monkeypatch.setattr("vllm_omni.diffusion.worker.diffusion_model_runner_v2.set_forward_context", _noop_forward_context)

        encode_payload = encode_runner.execute_encode(_make_engine_request("req-1", num_inference_steps=2))
        assert encode_payload.boundary == "encode_to_dit"

        transport_request = _make_engine_request("req-1", num_inference_steps=2)
        transport_request.stage_transport_payload = encode_payload
        first_step = dit_runner.execute_stepwise(_make_scheduler_output(transport_request))
        assert first_step.get_request_output("req-1").finished is False

        second_step = dit_runner.execute_stepwise(_make_cached_scheduler_output(request_id="req-1", step_id=1))
        dit_output = second_step.get_request_output("req-1")
        assert dit_output.finished is True
        dit_payload = dit_output.result.custom_output["stage_transport"]
        assert dit_payload.boundary == "dit_to_decode"

        output = decode_runner.execute_decode(dit_payload)
        assert output.error is None
        assert torch.equal(output.output, torch.tensor([2.0]))
        assert encode_runner.pipeline.prepare_calls == 1
        assert dit_runner.pipeline.denoise_calls == 2
        assert dit_runner.pipeline.scheduler_calls == 2
        assert decode_runner.pipeline.decode_calls == 1

class _RecordingLoRAManager:
    def __init__(self) -> None:
        self.calls: list[tuple[object | None, float]] = []

    def set_active_adapter(self, adapter, scale: float = 1.0) -> None:
        self.calls.append((adapter, scale))


def _make_step_worker(lora_manager=None, *, expected_output=None):
    """Build a bare DiffusionWorker primed for execute_stepwise tests."""
    worker = object.__new__(DiffusionWorker)
    worker.lora_manager = lora_manager
    worker._step_lora_state = {}
    output = expected_output if expected_output is not None else RunnerOutput(request_id="req-1")
    worker.model_runner = SimpleNamespace(execute_stepwise=lambda arg: output)
    return worker


@pytest.mark.cpu
class TestWorker:
    """DiffusionWorker.execute_stepwise"""

    def test_delegates_to_model_runner(self):
        expected = RunnerOutput(request_id="req-1", step_index=1, finished=False, result=None)
        worker = _make_step_worker(expected_output=expected)
        scheduler_output = _make_scheduler_output(_make_engine_request("req-1"), request_id="req-1")

        output = DiffusionWorker.execute_stepwise(worker, scheduler_output)

        assert output is expected

    def test_deactivates_lora_when_request_has_no_adapter(self):
        manager = _RecordingLoRAManager()
        worker = _make_step_worker(lora_manager=manager)
        scheduler_output = _make_scheduler_output(_make_engine_request("req-1"), request_id="req-1")

        DiffusionWorker.execute_stepwise(worker, scheduler_output)

        assert manager.calls == [(None, 1.0)]

    def test_activates_lora_for_step_requests(self):
        from vllm_omni.lora.request import LoRARequest

        lora_request = LoRARequest(lora_name="adapter", lora_int_id=7, lora_path="/tmp/lora")
        request = _make_engine_request("req-1")
        request.sampling_params.lora_request = lora_request
        request.sampling_params.lora_scale = 0.75

        manager = _RecordingLoRAManager()
        worker = _make_step_worker(lora_manager=manager)
        scheduler_output = _make_scheduler_output(request, request_id="req-1")

        DiffusionWorker.execute_stepwise(worker, scheduler_output)

        assert manager.calls == [(lora_request, 0.75)]

    def test_recovers_lora_for_cached_step_requests(self):
        from vllm_omni.lora.request import LoRARequest

        lora_request = LoRARequest(lora_name="adapter", lora_int_id=11, lora_path="/tmp/lora")
        request = _make_engine_request("req-1")
        request.sampling_params.lora_request = lora_request
        request.sampling_params.lora_scale = 0.5

        manager = _RecordingLoRAManager()
        worker = _make_step_worker(lora_manager=manager)
        first = _make_scheduler_output(request, request_id="req-1")
        second = _make_cached_scheduler_output(request_id="req-1", step_id=1)

        DiffusionWorker.execute_stepwise(worker, first)
        DiffusionWorker.execute_stepwise(worker, second)

        assert manager.calls == [(lora_request, 0.5), (lora_request, 0.5)]

    def test_activates_single_lora_for_homogeneous_batch(self):
        """Multiple requests sharing the same LoRA → exactly one activation,
        and every request id is registered in ``_step_lora_state``."""
        from vllm_omni.lora.request import LoRARequest

        lora_request = LoRARequest(lora_name="adapter", lora_int_id=9, lora_path="/tmp/lora")
        reqs = []
        for rid in ("req-1", "req-2", "req-3"):
            r = _make_engine_request(rid)
            r.sampling_params.lora_request = lora_request
            r.sampling_params.lora_scale = 0.6
            reqs.append(r)

        manager = _RecordingLoRAManager()
        worker = _make_step_worker(lora_manager=manager)
        scheduler_output = _make_batch_scheduler_output(reqs)

        DiffusionWorker.execute_stepwise(worker, scheduler_output)

        assert manager.calls == [(lora_request, 0.6)]
        assert set(worker._step_lora_state) == {"req-1", "req-2", "req-3"}
        for entry in worker._step_lora_state.values():
            assert entry == (lora_request, 0.6)

    def test_evicts_step_lora_state_for_finished_requests(self):
        from vllm_omni.lora.request import LoRARequest

        lora_request = LoRARequest(lora_name="adapter", lora_int_id=3, lora_path="/tmp/lora")
        finishing = _make_engine_request("req-1")
        finishing.sampling_params.lora_request = lora_request
        next_request = _make_engine_request("req-2")
        next_request.sampling_params.lora_request = lora_request

        worker = _make_step_worker(lora_manager=_RecordingLoRAManager())
        first = _make_scheduler_output(finishing, request_id="req-1")
        next_batch = _make_scheduler_output(
            next_request,
            request_id="req-2",
            step_id=1,
            finished_req_ids={"req-1"},
        )

        DiffusionWorker.execute_stepwise(worker, first)
        assert "req-1" in worker._step_lora_state

        DiffusionWorker.execute_stepwise(worker, next_batch)
        assert "req-1" not in worker._step_lora_state
        assert worker._step_lora_state == {"req-2": (lora_request, 1.0)}


@pytest.mark.cpu
class TestExecutor:
    """MultiprocDiffusionExecutor.execute_step"""

    def test_execute_step_passes_through_runner_output(self, mocker: MockerFixture):
        executor = object.__new__(MultiprocDiffusionExecutor)
        executor._ensure_open = lambda: None
        expected = RunnerOutput(request_id="req-step", step_index=1, finished=False, result=None)
        executor.collective_rpc = mocker.Mock(return_value=expected)

        request = _make_engine_request("req-step", num_inference_steps=2)
        scheduler_output = _make_scheduler_output(request, request_id="req-step")

        output = MultiprocDiffusionExecutor.execute_step(executor, scheduler_output)

        assert output is expected


@pytest.mark.cpu
class TestEngine:
    """Step-execution paths in DiffusionEngine.add_req_and_wait_for_response"""

    def _patch_engine_init(self, monkeypatch):
        monkeypatch.setattr(diffusion_engine_module, "get_diffusion_post_process_func", lambda _: None)
        monkeypatch.setattr(diffusion_engine_module, "get_diffusion_action_post_process_func", lambda _: None)
        monkeypatch.setattr(diffusion_engine_module, "get_diffusion_pre_process_func", lambda _: None)
        monkeypatch.setattr(DiffusionEngine, "_dummy_run", lambda self: None)

        class _Executor:
            def __init__(self, od_config):
                self.od_config = od_config

            def execute_step(self, scheduler_output):
                del scheduler_output
                return BatchRunnerOutput(outputs=[])

            def execute_request(self, request):
                del request
                return RunnerOutput(request_id="req-1", result=DiffusionOutput(output=torch.tensor([1.0])))

        monkeypatch.setattr(diffusion_engine_module.DiffusionExecutor, "get_class", lambda _: _Executor)

        return _Executor

    @pytest.mark.parametrize(
        ("step_execution", "stage_split"),
        [(True, False), (False, True)],
    )
    def test_stage_step_flags_must_match(self, monkeypatch, step_execution, stage_split):
        self._patch_engine_init(monkeypatch)
        config = SimpleNamespace(
            step_execution=step_execution,
            stage_split=stage_split,
            stage_role="dit",
            streaming_output=False,
        )

        with pytest.raises(ValueError, match="stage_split and step_execution"):
            DiffusionEngine(config)

    def test_stage_split_with_step_uses_step_scheduler(self, monkeypatch):
        executor_cls = self._patch_engine_init(monkeypatch)
        config = SimpleNamespace(
            step_execution=True,
            stage_split=True,
            stage_role="dit",
            streaming_output=False,
        )

        engine = DiffusionEngine(config)

        assert config.step_execution is True
        assert engine.step_execution is True
        assert engine.stepwise_execution is True
        assert isinstance(engine.scheduler, StepScheduler)
        assert engine.execute_fn.__self__ is engine.executor
        assert engine.execute_fn.__func__ is executor_cls.execute_step

    @pytest.mark.parametrize("stage_role", ["encode", "decode"])
    def test_split_encode_decode_roles_reject_scheduler_execution(self, monkeypatch, stage_role):
        self._patch_engine_init(monkeypatch)
        config = SimpleNamespace(
            step_execution=True,
            stage_split=True,
            stage_role=stage_role,
            streaming_output=False,
        )

        engine = DiffusionEngine(config)

        assert engine.stepwise_execution is False
        assert isinstance(engine.scheduler, RequestScheduler)
        with pytest.raises(RuntimeError, match=stage_role):
            engine.execute_fn(_make_scheduler_output(_make_step_request()))

    def test_normal_mode_uses_request_scheduler(self, monkeypatch):
        executor_cls = self._patch_engine_init(monkeypatch)
        config = SimpleNamespace(
            step_execution=False,
            stage_split=False,
            stage_role="all",
            streaming_output=False,
        )

        engine = DiffusionEngine(config)

        assert engine.stepwise_execution is False
        assert isinstance(engine.scheduler, RequestScheduler)
        assert engine.execute_fn.__self__ is engine.executor
        assert engine.execute_fn.__func__ is executor_cls.execute_request

    def test_non_split_rejects_stage_role(self, monkeypatch):
        self._patch_engine_init(monkeypatch)
        config = SimpleNamespace(
            step_execution=False,
            stage_split=False,
            stage_role="dit",
            streaming_output=False,
        )

        with pytest.raises(ValueError, match="stage_role must be 'all'"):
            DiffusionEngine(config)

    def test_split_rejects_all_role(self, monkeypatch):
        self._patch_engine_init(monkeypatch)
        config = SimpleNamespace(
            step_execution=True,
            stage_split=True,
            stage_role="all",
            streaming_output=False,
        )

        with pytest.raises(ValueError, match="stage_split=True requires stage_role"):
            DiffusionEngine(config)

    @pytest.mark.parametrize(
        ("execute_fn", "expected_error"),
        [
            (
                lambda _: RunnerOutput(
                    request_id="req-error",
                    step_index=1,
                    finished=True,
                    result=DiffusionOutput(error="boom"),
                ),
                "boom",
            ),
            (
                lambda _: (_ for _ in ()).throw(RuntimeError("gpu on fire")),
                "gpu on fire",
            ),
        ],
    )
    def test_step_engine_returns_error(self, execute_fn, expected_error, mocker: MockerFixture):
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace())
        engine = _make_engine(scheduler, execute_fn=execute_fn)

        output = engine.add_req_and_wait_for_response(_make_engine_request("req-error", num_inference_steps=2))

        assert output.output is None
        assert expected_error in output.error

    def test_step_execution_completes(self, mocker: MockerFixture):
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace())
        engine = _make_engine(scheduler)
        request = _make_engine_request("req-step", num_inference_steps=2)

        call_count = {"n": 0}

        def execute_fn(_):
            call_count["n"] += 1
            finished = call_count["n"] == 2
            return RunnerOutput(
                request_id="req-step",
                step_index=call_count["n"],
                finished=finished,
                result=(DiffusionOutput(output=torch.tensor([2.0])) if finished else None),
            )

        engine.execute_fn = execute_fn

        output = engine.add_req_and_wait_for_response(request)

        assert call_count["n"] == 2
        assert output.error is None
        assert torch.equal(output.output, torch.tensor([2.0]))

    def test_step_abort_stops_rescheduling_after_first_step(self, mocker: MockerFixture):
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace())
        engine = _make_engine(scheduler)
        request = _make_engine_request("req-stop", num_inference_steps=4)

        step = {"n": 0}

        def execute_fn(_):
            step["n"] += 1
            engine.abort("req-stop")
            return RunnerOutput(
                request_id="req-stop",
                step_index=1,
                finished=False,
                result=None,
            )

        engine.execute_fn = execute_fn

        output = engine.add_req_and_wait_for_response(request)

        assert step["n"] == 1
        _assert_aborted_output(output, "req-stop")

    def test_step_abort_after_reschedule_returns_aborted_output(self, mocker: MockerFixture):
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace())
        engine = _make_engine(scheduler)
        request = _make_engine_request("req-mid", num_inference_steps=4)

        step = {"n": 0}

        def execute_fn(sched_output):
            step["n"] += 1
            if step["n"] == 2:
                assert sched_output == _make_cached_scheduler_output("req-mid", step_id=1)
                engine.abort("req-mid")
            return RunnerOutput(
                request_id="req-mid",
                step_index=step["n"],
                finished=False,
                result=None,
            )

        engine.execute_fn = execute_fn

        output = engine.add_req_and_wait_for_response(request)

        assert step["n"] == 2
        _assert_aborted_output(output, "req-mid")

    def test_finished_step_without_result_returns_error(self, mocker: MockerFixture):
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace())
        engine = _make_engine(
            scheduler,
            execute_fn=lambda _: RunnerOutput(
                request_id="req-missing",
                step_index=1,
                finished=True,
                result=None,
            ),
        )

        output = engine.add_req_and_wait_for_response(_make_engine_request("req-missing", num_inference_steps=1))

        assert output.output is None
        assert output.error == "Diffusion execution finished without a final output."


@pytest.mark.cpu
class TestIPC:
    def test_pack_unpack_runner_output_shm(self):
        tensor = torch.zeros(300_000, dtype=torch.float32)
        output = RunnerOutput(request_id="req-1", finished=True, result=DiffusionOutput(output=tensor))

        packed = pack_diffusion_output_shm(output)
        assert isinstance(packed.result.output, dict)
        assert packed.result.output["__tensor_shm__"] is True

        unpacked = unpack_diffusion_output_shm(packed)
        assert isinstance(unpacked.result.output, torch.Tensor)
        torch.testing.assert_close(unpacked.result.output, tensor)


@pytest.mark.cpu
class TestSupportedPipelines:
    """Step-execution protocol checks for supported pipelines."""

    def test_top_level_step_execution_requires_stage_config(self):
        with pytest.raises(ValueError, match="step_execution=True requires"):
            AsyncOmniEngine._validate_diffusion_stage_step_configs(
                [],
                stage_configs_path=None,
                top_level_step_execution=True,
            )

    def test_diffusion_stage_config_requires_lockstep_fields(self):
        stage_cfg = SimpleNamespace(
            stage_type="diffusion",
            engine_args=SimpleNamespace(step_execution=True, stage_split=False, stage_role="all"),
        )

        with pytest.raises(ValueError, match="step_execution=True requires stage_split=True"):
            AsyncOmniEngine._validate_diffusion_stage_step_configs(
                [stage_cfg],
                stage_configs_path="dummy.yaml",
                top_level_step_execution=False,
            )

    def test_diffusion_stage_config_accepts_stage_step_roles(self):
        stage_cfgs = [
            SimpleNamespace(
                stage_type="diffusion",
                engine_args=SimpleNamespace(step_execution=True, stage_split=True, stage_role=role),
            )
            for role in ("encode", "dit", "decode")
        ]

        AsyncOmniEngine._validate_diffusion_stage_step_configs(
            stage_cfgs,
            stage_configs_path="dummy.yaml",
            top_level_step_execution=False,
        )

    def test_qwen_image_supports_step_execution(self):
        from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline

        # Avoid loading model weights; capability flags are class-level.
        pipeline = object.__new__(QwenImagePipeline)

        assert pipeline.supports_step_execution is True
        assert not hasattr(pipeline, "supports_stage_execution")

    def test_qwen_image_forward_keeps_mixin_entrypoint(self):
        from vllm_omni.diffusion.models.composed_pipeline import ComposedDiffusionPipeline
        from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline

        signature = inspect.signature(QwenImagePipeline.forward)

        assert QwenImagePipeline.forward is ComposedDiffusionPipeline.forward
        assert list(signature.parameters) == ["self", "req"]

    def test_stage_only_pipeline_is_not_a_valid_capability(self):
        from vllm_omni.diffusion.models.interface import supports_step_execution

        class _StageOnlyPipeline(_StepPipeline):
            supports_step_execution = False

        pipeline = _StageOnlyPipeline()

        assert supports_step_execution(pipeline) is False


@hardware_test(
    res={"cuda": "L4"},
    num_cards=2,
)
def test_execute_stepwise_with_ulysses_parallel():
    world_size = 2
    if current_omni_platform.get_device_count() < world_size:
        pytest.skip(f"Test requires {world_size} devices")

    torch.multiprocessing.spawn(
        _distributed_step_worker,
        args=(world_size, "ulysses", "29540"),
        nprocs=world_size,
    )


@hardware_test(
    res={"cuda": "L4"},
    num_cards=2,
)
def test_execute_stepwise_with_ring_parallel():
    world_size = 2
    if current_omni_platform.get_device_count() < world_size:
        pytest.skip(f"Test requires {world_size} devices")

    torch.multiprocessing.spawn(
        _distributed_step_worker,
        args=(world_size, "ring", "29541"),
        nprocs=world_size,
    )


@hardware_test(
    res={"cuda": "L4"},
    num_cards=2,
)
def test_execute_stepwise_with_cfg_parallel():
    world_size = 2
    if current_omni_platform.get_device_count() < world_size:
        pytest.skip(f"Test requires {world_size} devices")

    torch.multiprocessing.spawn(
        _distributed_step_worker,
        args=(world_size, "cfg", "29542"),
        nprocs=world_size,
    )
