# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for step-level diffusion execution across runner / worker / executor / engine."""

import asyncio
import contextlib
import os
import queue
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from pytest_mock import MockerFixture

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
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
    pack_stage_transport_shm,
    unpack_diffusion_output_shm,
    unpack_stage_transport_shm,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import StepScheduler
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionSchedulerOutput,
    NewRequestData,
)
from vllm_omni.diffusion.worker.input_batch import get_runner_step_config, set_runner_step_config
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.diffusion.worker.utils import DiffusionRequestState, DiffusionStateTransport, RunnerOutput
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
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

    supports_diffusion_atoms = True

    def __init__(self):
        self.prepare_calls = 0
        self.denoise_calls = 0
        self.scheduler_calls = 0
        self.decode_calls = 0

    def forward(self, req, **kwargs):
        del req, kwargs
        raise NotImplementedError

    def init_state(self, req):
        return DiffusionRequestState(
            request_id=req.request_id,
            sampling=req.sampling_params,
            prompts=req.prompts,
        )

    def validation(self, state):
        return state

    def encoding(self, state):
        return state

    def preparation(self, state):
        self.prepare_calls += 1
        num_steps = getattr(state.sampling, "num_inference_steps", None) or 2
        state.timesteps = [torch.tensor(10 - i) for i in range(num_steps)]
        state.latents = torch.tensor([0.0])
        return state

    def denoising(self, state):
        return state

    def predict_noise(self, input_batch):
        self.denoise_calls += 1
        return torch.full_like(input_batch.latents, fill_value=0.5)

    def advance_scheduler(self, state, noise_pred):
        del noise_pred
        self.scheduler_calls += 1
        state.latents = state.latents + 0.5
        state.step_index += 1
        return state

    def decoding(self, state):
        self.decode_calls += 1
        return state

    def postprocess(self, state):
        return DiffusionOutput(output=torch.tensor([state.step_index], dtype=torch.float32))

    def build_model_inputs(self, states):
        del states
        return {}

    def build_step_attention_metadata(self, batch):
        del batch
        return {}


class _InterruptingStepPipeline(_StepPipeline):
    interrupt = True

    def predict_noise(self, state):
        del state
        self.denoise_calls += 1
        return None

    def advance_scheduler(self, state, noise_pred):
        del state, noise_pred
        raise AssertionError("advance_scheduler should not run after interrupt")

    def decoding(self, state):
        del state
        raise AssertionError("decoding should not run after interrupt")


class _LegacyStepPipeline:
    """Old four-hook step protocol that should not enable stage-split."""


    def prepare_encode(self, state, **kwargs):
        del kwargs
        return state

    def denoise_step(self, input_batch, **kwargs):
        del input_batch, kwargs
        return None

    def step_scheduler(self, state, noise_pred, **kwargs):
        del state, noise_pred, kwargs

    def post_decode(self, state, **kwargs):
        del state, kwargs
        return DiffusionOutput()


class _IdentityNoiseTransformer(torch.nn.Module):
    def forward(self, x: torch.Tensor, **kwargs):
        del kwargs
        return (x,)


class _AdditiveScheduler:
    def step(self, noise_pred: torch.Tensor, t: torch.Tensor, latents: torch.Tensor, return_dict: bool = False):
        del t, return_dict
        return (latents + noise_pred,)


class _DistributedStepPipeline(CFGParallelMixin):
    supports_diffusion_atoms = True

    def __init__(self, mode: str, device: torch.device):
        self.mode = mode
        self.device = device
        self._interrupt = False
        self.scheduler = _AdditiveScheduler()
        self.transformer = _IdentityNoiseTransformer()

    @property
    def interrupt(self):
        return self._interrupt

    def forward(self, req, **kwargs):
        del req, kwargs
        raise NotImplementedError

    def init_state(self, req):
        return DiffusionRequestState(
            request_id=req.request_id,
            sampling=req.sampling_params,
            prompts=req.prompts,
        )

    def validation(self, state):
        return state

    def encoding(self, state):
        return state

    def preparation(self, state):
        state.timesteps = [torch.tensor(1.0, device=self.device)]
        state.latents = torch.ones((1, 1), device=self.device)
        state.step_index = 0
        state.scheduler = self.scheduler
        set_runner_step_config(state, do_true_cfg=self.mode == "cfg")
        return state

    def denoising(self, state):
        return state

    def predict_noise(self, input_batch=None, *args, **kwargs):
        if input_batch is None or not hasattr(input_batch, "latents"):
            if input_batch is None:
                return super().predict_noise(*args, **kwargs)
            return super().predict_noise(input_batch, *args, **kwargs)

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

    def advance_scheduler(self, state, noise_pred):
        do_true_cfg, _, _ = get_runner_step_config(state)
        if self.mode == "cfg":
            state.latents = self.scheduler_step_maybe_with_cfg(
                noise_pred,
                state.current_timestep,
                state.latents,
                do_true_cfg=do_true_cfg,
                per_request_scheduler=state.scheduler,
            )
        else:
            state.latents = state.latents + noise_pred
        state.step_index += 1
        return state

    def decoding(self, state):
        return state

    def postprocess(self, state):
        return DiffusionOutput(output=state.latents.detach().cpu())

    def build_model_inputs(self, states):
        del states
        return {}

    def build_step_attention_metadata(self, batch):
        del batch
        return {}


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


def _make_runner():
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = _make_vllm_config()
    runner.od_config = SimpleNamespace(
        cache_backend=None,
        parallel_config=SimpleNamespace(use_hsdp=False),
    )
    runner.device = torch.device("cpu")
    runner.pipeline = _StepPipeline()
    runner.cache_backend = None
    runner.offload_backend = None
    runner.state_cache = {}
    runner.kv_transfer_manager = SimpleNamespace()
    return runner


def _make_distributed_runner(mode: str, device: torch.device):
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = _make_vllm_config()
    runner.od_config = SimpleNamespace(
        cache_backend=None,
        parallel_config=SimpleNamespace(use_hsdp=False),
    )
    runner.device = device
    runner.pipeline = _DistributedStepPipeline(mode=mode, device=device)
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
    engine.od_config = SimpleNamespace(model_class_name="QwenImagePipeline", enable_cpu_offload=False)
    engine.pre_process_func = None
    engine.post_process_func = None
    engine._post_process_accepts_sampling_params = False
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

        runner = _make_distributed_runner(mode, device)
        result = DiffusionModelRunner.execute_stepwise(
            runner,
            _make_scheduler_output(_make_step_request(num_inference_steps=1), step_id=0),
        )
        output = result.get_request_output("req-1")

        assert output.finished is True
        assert output.result is not None
        transport = output.result.custom_output["stage_transport"]
        torch.testing.assert_close(
            transport.tensors["latents"].detach().cpu(),
            _expected_output_for_mode(mode).cpu(),
            rtol=1e-5,
            atol=1e-5,
        )
        assert "req-1" not in runner.state_cache
    finally:
        destroy_distributed_env()


# ---------------------------------------------------------------------------
# Runner / Worker
# ---------------------------------------------------------------------------


@pytest.mark.cpu
class TestRunner:
    """DiffusionModelRunner.execute_stepwise"""

    def test_completes_request_and_clears_state(self, monkeypatch):
        runner = _make_runner()
        req = _make_step_request()
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        result = DiffusionModelRunner.execute_stepwise(runner, _make_scheduler_output(req, step_id=0))
        first = result.get_request_output("req-1")
        assert first.request_id == "req-1"
        assert first.step_index == 1
        assert first.finished is False
        assert first.result is None
        assert "req-1" in runner.state_cache

        result = DiffusionModelRunner.execute_stepwise(runner, _make_cached_scheduler_output(step_id=1))
        second = result.get_request_output("req-1")
        assert second.request_id == "req-1"
        assert second.step_index == 2
        assert second.finished is True
        assert second.result is not None
        assert second.result.error is None
        transport = second.result.custom_output["stage_transport"]
        assert transport.boundary == "dit_to_decode"
        assert transport.meta["step_index"] == 2
        torch.testing.assert_close(transport.tensors["latents"], torch.tensor([1.0]))
        assert "req-1" not in runner.state_cache

        assert runner.pipeline.prepare_calls == 1
        assert runner.pipeline.denoise_calls == 2
        assert runner.pipeline.scheduler_calls == 2
        assert runner.pipeline.decode_calls == 0

    def test_denoiser_role_returns_decode_transport_without_reencoding(self, monkeypatch):
        encoder_runner = _make_runner()
        req = _make_step_request(num_inference_steps=1)
        encode_payload = DiffusionModelRunner.execute_encode(encoder_runner, req)
        assert encode_payload.boundary == "encode_to_dit"
        assert encoder_runner.pipeline.prepare_calls == 1

        denoiser_runner = _make_runner()
        denoiser_runner.od_config.diffusion_stage_role = "denoiser"
        denoise_req = _make_step_request(num_inference_steps=1)
        denoise_req.stage_transport_payload = encode_payload
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        result = DiffusionModelRunner.execute_stepwise(
            denoiser_runner,
            _make_scheduler_output(denoise_req, step_id=0),
        )

        output = result.get_request_output("req-1")
        assert output.finished is True
        assert output.result is not None
        transport = output.result.custom_output["stage_transport"]
        assert transport.boundary == "dit_to_decode"
        assert denoiser_runner.pipeline.prepare_calls == 0
        assert denoiser_runner.pipeline.denoise_calls == 1
        assert denoiser_runner.pipeline.scheduler_calls == 1
        assert denoiser_runner.pipeline.decode_calls == 0

    def test_decoder_role_consumes_decode_transport(self, monkeypatch):
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
        encoder_runner = _make_runner()
        encode_payload = DiffusionModelRunner.execute_encode(
            encoder_runner,
            _make_step_request(num_inference_steps=1),
        )
        denoiser_runner = _make_runner()
        denoiser_runner.od_config.diffusion_stage_role = "denoiser"
        denoise_req = _make_step_request(num_inference_steps=1)
        denoise_req.stage_transport_payload = encode_payload
        result = DiffusionModelRunner.execute_stepwise(
            denoiser_runner,
            _make_scheduler_output(denoise_req, step_id=0),
        )
        decode_payload = result.get_request_output("req-1").result.custom_output["stage_transport"]

        decoder_runner = _make_runner()
        decoder_runner.od_config.diffusion_stage_role = "decoder"
        output = DiffusionModelRunner.execute_decode(decoder_runner, decode_payload)

        assert torch.equal(output.output, torch.tensor([1.0]))
        assert decoder_runner.pipeline.decode_calls == 1

    def test_legacy_step_protocol_does_not_enable_stage_split(self, monkeypatch):
        from vllm_omni.diffusion.models.interface import supports_diffusion_atoms

        runner = _make_runner()
        runner.pipeline = _LegacyStepPipeline()
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        assert supports_diffusion_atoms(runner.pipeline) is False
        assert runner.supports_step_mode() is False
        with pytest.raises(ValueError, match="does not support step execution"):
            DiffusionModelRunner.execute_stepwise(
                runner,
                _make_scheduler_output(_make_step_request(num_inference_steps=1), step_id=0),
            )

    def test_rejects_multi_request_step_batch(self):
        runner = _make_runner()
        req_1 = _make_step_request()
        req_2 = _make_step_request()
        req_2.request_id = "req-2"

        scheduler_output = DiffusionSchedulerOutput(
            step_id=0,
            scheduled_new_reqs=[
                NewRequestData(request_id="req-1", req=req_1),
                NewRequestData(request_id="req-2", req=req_2),
            ],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            finished_req_ids=set(),
            num_running_reqs=2,
            num_waiting_reqs=0,
        )

        result = DiffusionModelRunner.execute_stepwise(runner, scheduler_output)
        assert len(result) == 2

    def test_rejects_missing_cached_state(self):
        runner = _make_runner()

        with pytest.raises(ValueError, match="Missing cached state"):
            DiffusionModelRunner.execute_stepwise(runner, _make_cached_scheduler_output(request_id="req-missing"))

    def test_interrupt_marks_request_finished_and_clears_state(self, monkeypatch):
        runner = _make_runner()
        runner.pipeline = _InterruptingStepPipeline()
        req = _make_step_request()
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        result = DiffusionModelRunner.execute_stepwise(runner, _make_scheduler_output(req, step_id=0))
        output = result.get_request_output("req-1")
        assert output.request_id == "req-1"
        assert output.step_index == 0
        assert output.finished is True
        assert output.result is not None
        assert output.result.error == "stepwise denoise interrupted"
        assert "req-1" not in runner.state_cache
        assert runner.pipeline.prepare_calls == 1
        assert runner.pipeline.denoise_calls == 1
        assert runner.pipeline.scheduler_calls == 0
        assert runner.pipeline.decode_calls == 0

    def test_load_model_rejects_unsupported_step_execution(self, monkeypatch):
        class _RequestOnlyPipeline:
            pass

        class _FakeLoader:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def load_model(self, **kwargs):
                del kwargs
                return _RequestOnlyPipeline()

        class _FakeProfiler:
            consumed_memory = 0

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

        runner = object.__new__(DiffusionModelRunner)
        runner.vllm_config = _make_vllm_config()
        runner.od_config = SimpleNamespace(
            enable_cpu_offload=False,
            enable_layerwise_offload=False,
            enforce_eager=True,
            cache_backend=None,
            cache_config=None,
            step_execution=True,
            diffusion_stage_role="denoiser",
            model_class_name="RequestOnlyPipeline",
            parallel_config=SimpleNamespace(use_hsdp=False),
        )
        runner.device = torch.device("cpu")
        runner.pipeline = None
        runner.cache_backend = None
        runner.offload_backend = None
        runner.state_cache = {}
        runner.kv_transfer_manager = SimpleNamespace()

        monkeypatch.setattr(model_runner_module, "DiffusersPipelineLoader", _FakeLoader)
        monkeypatch.setattr(model_runner_module, "DeviceMemoryProfiler", _FakeProfiler)
        monkeypatch.setattr(model_runner_module, "get_offload_backend", lambda *args, **kwargs: None)
        monkeypatch.setattr(model_runner_module, "get_cache_backend", lambda *args, **kwargs: None)

        with pytest.raises(ValueError, match="RequestOnlyPipeline"):
            DiffusionModelRunner.load_model(runner)


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

    def test_postprocess_diffusion_output_uses_engine_postprocess(self):
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace())
        engine = _make_engine(scheduler)
        engine.post_process_func = lambda output: ["converted-image", output.tolist()]
        request = _make_engine_request("req-post", num_inference_steps=1)

        results = engine.postprocess_diffusion_output(
            request,
            DiffusionOutput(output=torch.tensor([1.0])),
            exec_total_time=0.001,
        )

        assert len(results) == 1
        assert results[0].images == ["converted-image", [1.0]]
        assert results[0].metrics["diffusion_engine_exec_time_ms"] == 1.0

    def test_postprocess_diffusion_output_skips_image_postprocess_for_latents(self):
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace())
        engine = _make_engine(scheduler)

        def _unexpected_postprocess(output):
            del output
            raise AssertionError("latent output must not run image postprocess")

        engine.post_process_func = _unexpected_postprocess
        request = _make_engine_request("req-latent", num_inference_steps=1)
        request.sampling_params.output_type = "latent"
        latents = torch.ones((1, 4, 8), dtype=torch.float32)

        results = engine.postprocess_diffusion_output(
            request,
            DiffusionOutput(output=latents),
            exec_total_time=0.001,
        )

        assert len(results) == 1
        assert results[0].images == []
        assert results[0].final_output_type == "latents"
        torch.testing.assert_close(results[0].latents, latents)
        assert results[0].metrics["diffusion_engine_exec_time_ms"] == 1.0


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

    def test_pack_unpack_stage_transport_shm_preserves_tensor_tree(self):
        sampling = OmniDiffusionSamplingParams(num_inference_steps=2, output_type="pil")
        state = DiffusionRequestState(
            request_id="req-transport",
            sampling=sampling,
            prompts=["prompt"],
            latents=torch.ones((1, 4), dtype=torch.float32),
            timesteps=[torch.tensor(9.0), torch.tensor(3.0)],
            step_index=1,
        )
        state.conditioning = {
            "prompt_embeds": torch.zeros((300_000,), dtype=torch.float32),
            "nested": [torch.tensor([1.0])],
        }

        payload = state.to_transport("encode_to_dit", conditioning=state.conditioning)
        packed = pack_stage_transport_shm(payload)
        assert packed.conditioning["prompt_embeds"]["__tensor_shm__"] is True

        unpacked = unpack_stage_transport_shm(packed)
        restored = DiffusionRequestState.from_transport(unpacked)

        assert restored.request_id == "req-transport"
        assert restored.step_index == 1
        assert restored.sampling.output_type == "pil"
        torch.testing.assert_close(restored.latents, state.latents)
        torch.testing.assert_close(restored.timesteps[0], torch.tensor(9.0))
        torch.testing.assert_close(restored.conditioning["prompt_embeds"], state.conditioning["prompt_embeds"])
        torch.testing.assert_close(restored.conditioning["nested"][0], torch.tensor([1.0]))

    def test_qwen_stage_transport_is_msgpack_safe_and_keeps_runner_cfg(self):
        from vllm_omni.diffusion.models.qwen_image.step_batch import pack_qwen_conditioning, set_qwen_state
        from vllm_omni.distributed.omni_connectors.utils.serialization import OmniMsgpackEncoder

        state = DiffusionRequestState(
            request_id="req-qwen-transport",
            sampling=OmniDiffusionSamplingParams(num_inference_steps=2, output_type="latent"),
            prompts=[
                {
                    "prompt": "prompt",
                    "negative_prompt": " ",
                    "multi_modal_data": {"image": object()},
                    "additional_information": {
                        "prompt_image": object(),
                        "preprocessed_image": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
                    },
                }
            ],
            latents=torch.ones((1, 4), dtype=torch.float32),
            timesteps=torch.tensor([9.0, 3.0]),
            step_index=0,
        )
        set_runner_step_config(state, do_true_cfg=True, true_cfg_scale=4.0, cfg_normalize=True)
        state.sampling.generator = torch.Generator(device="cpu").manual_seed(456)
        set_qwen_state(
            state,
            atom_args={
                "prompt": "prompt",
                "height": 512,
                "width": 512,
                "layers": 3,
                "generator": torch.Generator(device="cpu").manual_seed(123),
                "image": object(),
            },
            prompt_embeds=torch.zeros((1, 4, 8), dtype=torch.float32),
            prompt_embeds_mask=torch.ones((1, 4), dtype=torch.bool),
            txt_seq_lens=[4],
            img_shapes=[[(1, 32, 32)]],
        )

        payload = state.to_transport("encode_to_dit", conditioning=pack_qwen_conditioning(state))
        assert "generator" not in payload.conditioning["atom_args"]
        assert "image" not in payload.conditioning["atom_args"]
        assert payload.conditioning["atom_args"]["layers"] == 3

        packed = pack_stage_transport_shm(payload)
        OmniMsgpackEncoder().encode(packed)
        restored = DiffusionRequestState.from_transport(unpack_stage_transport_shm(packed))
        assert restored.prompts == [{"prompt": "prompt", "negative_prompt": " "}]
        assert get_runner_step_config(restored) == (True, 4.0, True)

    def test_qwen_model_inputs_keep_multirow_seq_lens_and_shapes(self):
        from vllm_omni.diffusion.models.qwen_image.step_batch import build_qwen_model_inputs, set_qwen_state

        state = DiffusionRequestState(
            request_id="req-qwen-batch",
            sampling=OmniDiffusionSamplingParams(num_outputs_per_prompt=2),
            prompts=["prompt"],
        )
        set_qwen_state(
            state,
            prompt_embeds=torch.zeros((2, 4, 8), dtype=torch.float32),
            prompt_embeds_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool),
            txt_seq_lens=[4, 3],
            img_shapes=[[(1, 32, 32)], [(1, 32, 32)]],
        )

        model_inputs = build_qwen_model_inputs([state])

        assert model_inputs["prompt_embeds"].shape[0] == 2
        assert model_inputs["txt_seq_lens"] == [4, 3]
        assert model_inputs["img_shapes"] == [[(1, 32, 32)], [(1, 32, 32)]]

    def test_qwen_model_inputs_use_request_attention_kwargs(self):
        from vllm_omni.diffusion.models.qwen_image.step_batch import build_qwen_model_inputs, set_qwen_state

        state = DiffusionRequestState(
            request_id="req-qwen-attn",
            sampling=OmniDiffusionSamplingParams(),
            prompts=["prompt"],
        )
        set_qwen_state(
            state,
            prompt_embeds=torch.zeros((1, 4, 8), dtype=torch.float32),
            prompt_embeds_mask=torch.ones((1, 4), dtype=torch.bool),
            txt_seq_lens=[4],
            attention_kwargs={"scale": 0.5},
        )

        model_inputs = build_qwen_model_inputs([state])

        assert model_inputs["extra_transformer_kwargs"]["attention_kwargs"] == {"scale": 0.5}

    def test_dit_to_decode_transport_drops_conditioning(self):
        state = DiffusionRequestState(
            request_id="req-decode-boundary",
            sampling=OmniDiffusionSamplingParams(),
            prompts=["prompt"],
            conditioning={"prompt_embeds": torch.ones((1, 2), dtype=torch.float32)},
            latents=torch.ones((1, 4), dtype=torch.float32),
        )

        payload = state.to_transport("dit_to_decode", conditioning=None)

        assert payload.conditioning is None
        assert payload.extra == {}
        torch.testing.assert_close(payload.tensors["latents"], state.latents)

    def test_qwen_predict_noise_preserves_cfg_transformer_kwargs_path(self):
        from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
        from vllm_omni.diffusion.models.qwen_image.step_batch import QwenImageStepAtomsMixin

        class _QwenMixinPipeline(QwenImageStepAtomsMixin, CFGParallelMixin):
            def __init__(self):
                self.transformer = _IdentityNoiseTransformer()

        x = torch.tensor([1.0])
        base_pipeline = object.__new__(QwenImagePipeline)
        torch.nn.Module.__init__(base_pipeline)
        base_pipeline.transformer = _IdentityNoiseTransformer()

        torch.testing.assert_close(base_pipeline.predict_noise(x=x), x)
        torch.testing.assert_close(_QwenMixinPipeline().predict_noise(x=x), x)

    def test_qwen_decode_accepts_latents_alias_without_vae_decode(self):
        from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
        from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_layered import QwenImageLayeredPipeline
        from vllm_omni.diffusion.models.qwen_image.step_batch import QwenImageStepAtomsMixin

        class _MixinOnly(QwenImageStepAtomsMixin):
            pass

        latents = torch.ones((2, 4, 8), dtype=torch.float32)
        for pipeline in (
            object.__new__(QwenImagePipeline),
            object.__new__(QwenImageLayeredPipeline),
            object.__new__(_MixinOnly),
        ):
            output = pipeline._decode_latents(latents, height=64, width=64, output_type="latents")
            torch.testing.assert_close(output.output, latents)

    def test_merge_latent_outputs_keeps_all_prompts(self):
        from vllm_omni.diffusion.stage_merge import merge_latent_outputs

        first = torch.ones((1, 4, 8), dtype=torch.float32)
        second = torch.full((2, 4, 8), 2.0, dtype=torch.float32)

        merged = merge_latent_outputs([first, None, second])

        assert merged.shape == (3, 4, 8)
        torch.testing.assert_close(merged[0], first[0])
        torch.testing.assert_close(merged[1:], second)

    def test_qwen_stage_role_trimming_drops_unused_modules(self):
        from vllm_omni.diffusion.models.qwen_image.step_batch import configure_qwen_stage_role

        pipeline = SimpleNamespace(
            text_encoder=object(),
            tokenizer=object(),
            processor=object(),
            vl_processor=object(),
            transformer=object(),
            scheduler=object(),
            vae=object(),
            image_processor=object(),
        )

        dropped = configure_qwen_stage_role(pipeline, "decode")

        assert set(dropped) == {
            "text_encoder",
            "tokenizer",
            "processor",
            "vl_processor",
            "transformer",
        }
        assert hasattr(pipeline, "vae")
        assert hasattr(pipeline, "scheduler")
        assert hasattr(pipeline, "image_processor")

    def test_runner_calls_pipeline_stage_role_config(self, mocker: MockerFixture):
        runner = object.__new__(DiffusionModelRunner)
        runner.od_config = SimpleNamespace(diffusion_stage_role="dit")
        calls = []

        class _Pipeline:
            def configure_diffusion_stage_role(self, role):
                calls.append(role)

        runner.pipeline = _Pipeline()
        mocker.patch.object(model_runner_module.current_omni_platform, "empty_cache", lambda: None)

        runner._configure_pipeline_stage_role()

        assert calls == ["dit"]

    def test_transport_conditioning_is_unpacked_after_device_move(self):
        captured: dict[str, Any] = {}

        class _Pipeline:
            def unpack_conditioning(self, payload, state):
                captured["payload"] = payload
                state.conditioning = payload
                return state

        runner = object.__new__(DiffusionModelRunner)
        runner.pipeline = _Pipeline()
        runner.device = torch.device("meta")
        payload = DiffusionStateTransport(
            boundary="encode_to_dit",
            request_id="req-device",
            sampling=OmniDiffusionSamplingParams(),
            prompts=["prompt"],
            conditioning={"prompt_embeds": torch.ones((1, 2), dtype=torch.float32)},
        )

        state = runner._state_from_transport(payload, expected_boundary="encode_to_dit")

        assert captured["payload"]["prompt_embeds"].device.type == "meta"
        assert state.conditioning["prompt_embeds"].device.type == "meta"

    def test_stage_pool_treats_direct_latents_as_non_empty_output(self):
        from vllm_omni.engine.stage_pool import StagePool

        pool = StagePool(
            stage_id=0,
            clients=SimpleNamespace(
                final_output_type="latents",
                final_output=False,
                stage_type="diffusion",
            ),
        )
        request_output = SimpleNamespace(
            final_output_type="latents",
            latents=torch.ones((1, 4, 8), dtype=torch.float32),
            trajectory_latents=None,
            images=[],
            outputs=[],
        )

        assert pool.has_non_empty_output(request_output) is True

    def test_stage_pool_treats_stage_transport_as_empty_output(self):
        from vllm_omni.engine.stage_pool import StagePool

        pool = StagePool(
            stage_id=0,
            clients=SimpleNamespace(
                final_output_type="stage_transport",
                final_output=False,
                stage_type="diffusion",
            ),
        )
        request_output = StagePool._make_stage_transport_output(
            "req-transport",
            {"latents": torch.ones((1, 4), dtype=torch.float32)},
        )

        assert pool.has_non_empty_output(request_output) is False

    @pytest.mark.asyncio
    async def test_stage_pool_abort_cancels_inflight_diffusion_task(self):
        from vllm_omni.engine.stage_pool import StagePool

        class _Client:
            stage_type = "diffusion"
            final_output = False
            final_output_type = "stage_transport"
            diffusion_stage_role = "denoiser"

            def __init__(self):
                self.abort_calls = []

            async def abort_requests_async(self, request_ids):
                self.abort_calls.append(list(request_ids))

        client = _Client()
        pool = StagePool(stage_id=0, clients=[client])
        pool._request_bindings["req-abort"] = 0
        task = asyncio.create_task(asyncio.sleep(60))
        pool._diffusion_stage_tasks["req-abort"] = task

        await pool.abort_requests(["req-abort"])
        await asyncio.sleep(0)

        assert task.cancelled()
        assert "req-abort" not in pool._diffusion_stage_tasks
        assert client.abort_calls == [["req-abort"]]

    @pytest.mark.asyncio
    async def test_stage_pool_encoder_role_accepts_batched_prompts(self):
        from vllm_omni.engine.stage_pool import StagePool

        class _Client:
            def __init__(self):
                self.batch_calls = []
                self.outputs = []

            async def stage_encode_request_async(self, *args, **kwargs):
                raise AssertionError("single-prompt encoder path should not be used")

            async def stage_encode_batch_request_async(
                self,
                request_id,
                prompts,
                sampling_params,
                *,
                kv_sender_info=None,
            ):
                self.batch_calls.append((request_id, list(prompts), sampling_params, kv_sender_info))
                return {"prompts": list(prompts)}

            def put_diffusion_output_nowait(self, output):
                self.outputs.append(output)

        pool = object.__new__(StagePool)
        pool.stage_id = 0
        pool._diffusion_stage_tasks = {}
        client = _Client()
        sampling_params = OmniDiffusionSamplingParams()
        kv_sender_info = {0: {"host": "127.0.0.1", "zmq_port": 50151}}

        await pool._run_diffusion_stage_role(
            "req-batch",
            client,
            "encoder",
            ["prompt-1", "prompt-2"],
            sampling_params,
            kv_sender_info=kv_sender_info,
        )

        assert client.batch_calls == [("req-batch", ["prompt-1", "prompt-2"], sampling_params, kv_sender_info)]
        assert client.outputs[0].request_id == "req-batch"
        assert client.outputs[0].custom_output["stage_transport"] == {"prompts": ["prompt-1", "prompt-2"]}
        assert client.outputs[0].final_output_type == "stage_transport"


@pytest.mark.cpu
class TestSupportedPipelines:
    """Step-execution protocol checks for supported pipelines."""

    def test_default_stage_config_keeps_monolithic_on_forward_path(self):
        stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
            {
                "step_execution": True,
            }
        )[0]

        assert stage_cfg["engine_args"]["step_execution"] is False
        assert stage_cfg["engine_args"]["diffusion_stage_role"] == "monolithic"

    def test_default_stage_config_enables_step_execution_for_split_role(self):
        stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
            {
                "diffusion_stage_role": "denoiser",
            }
        )[0]

        assert stage_cfg["engine_args"]["step_execution"] is True
        assert stage_cfg["engine_args"]["diffusion_stage_role"] == "denoiser"

    def test_default_stage_config_maps_model_stage_alias_to_split_role(self):
        stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
            {
                "model_stage": "dit",
            }
        )[0]

        assert stage_cfg["engine_args"]["step_execution"] is True
        assert stage_cfg["engine_args"]["diffusion_stage_role"] == "denoiser"
        assert stage_cfg["engine_args"]["model_stage"] == "dit"

    def test_diffusion_stage_role_mapping(self):
        from vllm_omni.diffusion.stage_kind import DiffusionStageKind, diffusion_role_from_model_stage, role_affinity

        assert diffusion_role_from_model_stage("diffusion").value == "monolithic"
        assert diffusion_role_from_model_stage("encode").value == "encoder"
        assert diffusion_role_from_model_stage("dit").value == "denoiser"
        assert diffusion_role_from_model_stage("decode").value == "decoder"
        assert role_affinity("denoiser", DiffusionStageKind.DENOISING) is True
        assert role_affinity("denoiser", DiffusionStageKind.DECODING) is False

    def test_split_diffusion_stage_skips_monolithic_dummy_run(self):
        engine = object.__new__(DiffusionEngine)
        engine.od_config = SimpleNamespace(diffusion_stage_role="encoder")

        engine._dummy_run()

    def test_orchestrator_forwards_stage_transport_as_diffusion_prompt(self):
        from vllm_omni.engine.orchestrator import Orchestrator
        from vllm_omni.engine.stage_pool import StagePool

        payload = {"latents": torch.tensor([1.0])}
        output = StagePool._make_stage_transport_output("req-transport", payload)

        prompt = Orchestrator._stage_transport_prompt_from_output(output)

        assert prompt is not None
        torch.testing.assert_close(prompt["stage_transport"]["latents"], payload["latents"])

    @pytest.mark.parametrize(
        "pipeline_cls_path",
        [
            "vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image.QwenImagePipeline",
            "vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit.QwenImageEditPipeline",
            "vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit_plus.QwenImageEditPlusPipeline",
            "vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_layered.QwenImageLayeredPipeline",
        ],
    )
    def test_qwen_image_supports_diffusion_atoms(self, pipeline_cls_path):
        import importlib

        from vllm_omni.diffusion.models.interface import (
            SupportsDiffusionAtoms,
            supports_diffusion_atoms,
        )

        module_name, cls_name = pipeline_cls_path.rsplit(".", 1)
        pipeline_cls = getattr(importlib.import_module(module_name), cls_name)
        # Avoid loading model weights; protocol membership depends on the class contract.
        pipeline = object.__new__(pipeline_cls)

        assert pipeline.supports_diffusion_atoms is True
        assert supports_diffusion_atoms(pipeline) is True
        assert isinstance(pipeline, SupportsDiffusionAtoms) is True
        assert getattr(pipeline, "interrupt", False) is False
        assert getattr(pipeline, "attention_kwargs", None) == {}
        assert getattr(pipeline, "current_timestep", "unset") is None

    @pytest.mark.parametrize(
        "pipeline_cls_path",
        [
            "vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image.QwenImagePipeline",
            "vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit.QwenImageEditPipeline",
            "vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit_plus.QwenImageEditPlusPipeline",
            "vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_layered.QwenImageLayeredPipeline",
        ],
    )
    def test_qwen_image_forward_stays_upstream_shaped(self, pipeline_cls_path):
        import importlib
        import inspect

        module_name, cls_name = pipeline_cls_path.rsplit(".", 1)
        pipeline_cls = getattr(importlib.import_module(module_name), cls_name)

        source = inspect.getsource(pipeline_cls.forward)

        assert "run_qwen_atom_forward" not in source
        assert "req.sampling_params.true_cfg_scale or true_cfg_scale" not in source
        assert "req.sampling_params.true_cfg_scale is not None" in source
        assert "req.sampling_params.output_type is not None" in source
        if cls_name == "QwenImageLayeredPipeline":
            assert "images = image" in source

    @pytest.mark.parametrize(
        "pipeline_cls_path",
        [
            "vllm_omni.diffusion.models.qwen_image.step_batch.QwenImageStepAtomsMixin",
            "vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image.QwenImagePipeline",
            "vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_layered.QwenImageLayeredPipeline",
        ],
    )
    def test_qwen_rehydrate_restores_transport_attention_kwargs(self, pipeline_cls_path):
        import importlib

        from vllm_omni.diffusion.models.qwen_image.step_batch import set_qwen_state

        module_name, cls_name = pipeline_cls_path.rsplit(".", 1)
        pipeline_cls = getattr(importlib.import_module(module_name), cls_name)
        pipeline = object.__new__(pipeline_cls)
        object.__setattr__(pipeline, "scheduler", None)
        state = DiffusionRequestState(
            request_id="req-attention",
            sampling=OmniDiffusionSamplingParams(),
        )
        set_qwen_state(state, atom_args={"attention_kwargs": {"scale": 0.5}})

        pipeline.rehydrate_stage_state(state)

        assert getattr(pipeline, "_attention_kwargs") == {"scale": 0.5}


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
