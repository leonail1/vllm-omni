# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.utils import DiffusionRequestState

pytestmark = [pytest.mark.diffusion]


@contextmanager
def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    yield


class _DummyPipeline:
    def __init__(self, output):
        self._output = output
        self.forward_calls = 0

    def forward(self, req):
        del req
        self.forward_calls += 1
        return self._output


def _make_request(skip_cache_refresh: bool = True):
    sampling_params = SimpleNamespace(
        generator=None,
        seed=None,
        generator_device=None,
        num_inference_steps=4,
    )
    return SimpleNamespace(
        prompts=["a prompt"],
        sampling_params=sampling_params,
        skip_cache_refresh=skip_cache_refresh,
    )


def _make_runner(cache_backend, cache_backend_name: str, enable_cache_dit_summary: bool = True):
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.pipeline = _DummyPipeline(output=SimpleNamespace(output="ok"))
    runner.cache_backend = cache_backend
    runner.offload_backend = None
    runner.od_config = SimpleNamespace(
        cache_backend=cache_backend_name,
        enable_cache_dit_summary=enable_cache_dit_summary,
        parallel_config=SimpleNamespace(use_hsdp=False),
    )
    runner.kv_transfer_manager = SimpleNamespace(
        receive_kv_cache=lambda req, target_device=None: None,
        receive_multi_kv_cache=lambda req, cfg_kv_collect_func=None, target_device=None: None,
        receive_multi_kv_cache_distributed=lambda req, cfg_kv_collect_func=None, target_device=None: None,
    )
    return runner


def _make_stepwise_runner(tmp_path):
    class _StepPipeline:
        interrupt = False
        input_batch_during_decode = "unset"
        runner = None

        def denoise_step(self, input_batch):
            return torch.ones(input_batch.num_reqs, 1)

        def step_scheduler(self, req, noise_pred):
            del noise_pred
            req.step_index += 1

        def post_decode(self, req):
            if self.runner is not None:
                self.input_batch_during_decode = self.runner.input_batch
            return SimpleNamespace(decoded=req.req_id)

    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.pipeline = _StepPipeline()
    runner.pipeline.runner = runner
    runner.cache_backend = None
    runner.offload_backend = None
    runner.state_cache = {}
    runner.od_config = SimpleNamespace(
        model="Qwen/Qwen-Image",
        stage_id=0,
        cache_backend="none",
        max_num_seqs=2,
        omni_replica_id=0,
        parallel_config=SimpleNamespace(use_hsdp=False, tensor_parallel_size=2),
        additional_config={
            "diffusion_step_profile": {
                "enabled": True,
                "output_path": str(tmp_path / "step_cost_raw.jsonl"),
                "sync_device": False,
            }
        },
    )
    runner.step_cost_profiler = model_runner_module.DiffusionStepCostProfiler.from_od_config(
        runner.od_config,
        device=runner.device,
    )
    runner.supports_step_mode = lambda: True
    return runner


def test_execute_stepwise_emits_step_cost_profile_record(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    runner = _make_stepwise_runner(tmp_path)
    state = DiffusionRequestState(
        req_id="req-0",
        sampling=SimpleNamespace(
            height=512,
            width=512,
            num_frames=1,
            num_outputs_per_prompt=1,
            do_classifier_free_guidance=False,
            guidance_scale=0.0,
            true_cfg_scale=None,
            extra_args={"profile_combo_id": "512x512_b1", "profile_phase": "measure"},
        ),
        latents=torch.zeros(1, 1),
        timesteps=torch.arange(1),
        step_index=0,
    )
    scheduler_output = SimpleNamespace(
        step_id=3,
        scheduled_req_ids=["req-0"],
    )

    runner._update_states = lambda output: ([state], [])
    runner._prepare_batch_inputs = lambda states, new_request_ids: SimpleNamespace(num_reqs=len(states))
    runner._prepare_attn_metadata = lambda input_batch: {}
    def _fail_update_states_after(states, input_batch, interrupted):
        del states, input_batch, interrupted
        raise AssertionError("final decode should skip cached batch refresh")

    runner.state_cache["req-0"] = state
    runner._update_states_after = _fail_update_states_after
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

    output = DiffusionModelRunner.execute_stepwise(runner, scheduler_output)

    assert output.get_req_output("req-0").finished is True
    assert runner.pipeline.input_batch_during_decode is None
    assert runner.state_cache == {}
    rows = [json.loads(line) for line in (tmp_path / "step_cost_raw.jsonl").read_text().splitlines()]
    assert rows[0]["batch_size"] == 1
    assert rows[0]["denoise_ms"] >= 0
    assert rows[0]["step_scheduler_ms"] >= 0
    assert rows[0]["post_decode_ms"] >= 0


def test_execute_stepwise_cleans_state_cache_when_post_decode_fails(tmp_path, monkeypatch):
    runner = _make_stepwise_runner(tmp_path)
    state = DiffusionRequestState(
        req_id="req-0",
        sampling=SimpleNamespace(
            height=512,
            width=512,
            num_frames=1,
            num_outputs_per_prompt=1,
            do_classifier_free_guidance=False,
            guidance_scale=0.0,
            true_cfg_scale=None,
            extra_args={"profile_combo_id": "512x512_b1", "profile_phase": "measure"},
        ),
        latents=torch.zeros(1, 1),
        timesteps=torch.arange(1),
        step_index=0,
    )
    scheduler_output = SimpleNamespace(
        step_id=3,
        scheduled_req_ids=["req-0"],
    )

    runner.state_cache["req-0"] = state
    runner._update_states = lambda output: ([state], [])
    runner._prepare_batch_inputs = lambda states, new_request_ids: SimpleNamespace(num_reqs=len(states))
    runner._prepare_attn_metadata = lambda input_batch: {}
    runner._update_states_after = lambda states, input_batch, interrupted: None

    def _raise_post_decode(req):
        del req
        raise RuntimeError("decode failed")

    runner.pipeline.post_decode = _raise_post_decode
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

    with pytest.raises(RuntimeError, match="decode failed"):
        DiffusionModelRunner.execute_stepwise(runner, scheduler_output)

    assert runner.input_batch is None
    assert runner.state_cache == {}


def test_execute_stepwise_skips_disabled_step_cost_profiler(tmp_path, monkeypatch):
    class _DisabledProfiler:
        enabled = False

        def timer_start(self):
            raise AssertionError("disabled profiler timer should not be called")

        def timer_end_ms(self, start_s):
            del start_s
            raise AssertionError("disabled profiler timer should not be called")

        def write_step_record(self, **kwargs):
            del kwargs
            raise AssertionError("disabled profiler writer should not be called")

    runner = _make_stepwise_runner(tmp_path)
    runner.step_cost_profiler = _DisabledProfiler()
    state = DiffusionRequestState(
        req_id="req-0",
        sampling=SimpleNamespace(
            height=512,
            width=512,
            num_frames=1,
            num_outputs_per_prompt=1,
            do_classifier_free_guidance=False,
            guidance_scale=0.0,
            true_cfg_scale=None,
            extra_args={"profile_combo_id": "512x512_b1", "profile_phase": "measure"},
        ),
        latents=torch.zeros(1, 1),
        timesteps=torch.arange(1),
        step_index=0,
    )
    scheduler_output = SimpleNamespace(
        step_id=3,
        scheduled_req_ids=["req-0"],
    )

    runner._update_states = lambda output: ([state], [])
    runner._prepare_batch_inputs = lambda states, new_request_ids: SimpleNamespace(num_reqs=len(states))
    runner._prepare_attn_metadata = lambda input_batch: {}
    runner._update_states_after = lambda states, input_batch, interrupted: None
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

    output = DiffusionModelRunner.execute_stepwise(runner, scheduler_output)

    assert output.get_req_output("req-0").finished is True
    assert not (tmp_path / "step_cost_raw.jsonl").exists()


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_execute_model_skips_cache_summary_without_active_cache_backend(monkeypatch):
    """Guard cache diagnostics with runtime backend state to avoid stale-config crashes."""
    runner = _make_runner(cache_backend=None, cache_backend_name="cache_dit")
    req = _make_request(skip_cache_refresh=True)

    cache_summary_calls = []

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(
        model_runner_module,
        "cache_summary",
        lambda pipeline, details: cache_summary_calls.append((pipeline, details)),
    )

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.output == "ok"
    assert cache_summary_calls == []


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_execute_model_emits_cache_summary_with_active_cache_dit_backend(monkeypatch):
    class _EnabledCacheBackend:
        def is_enabled(self):
            return True

    runner = _make_runner(cache_backend=_EnabledCacheBackend(), cache_backend_name="cache_dit")
    req = _make_request(skip_cache_refresh=True)

    cache_summary_calls = []

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(
        model_runner_module,
        "cache_summary",
        lambda pipeline, details: cache_summary_calls.append((pipeline, details)),
    )

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.output == "ok"
    assert cache_summary_calls == [(runner.pipeline, True)]


@pytest.mark.core_model
@pytest.mark.cpu
def test_load_model_clears_cache_backend_for_unsupported_pipeline(monkeypatch):
    class _DummyLoader:
        def __init__(self, load_config, od_config=None):
            del load_config, od_config

        def load_model(self, **kwargs):
            del kwargs
            return SimpleNamespace(transformer=torch.nn.Identity())

    class _DummyMemoryProfiler:
        consumed_memory = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    class _DummyCacheBackend:
        def __init__(self):
            self.enabled = False

        def enable(self, pipeline):
            del pipeline
            self.enabled = True

    dummy_cache_backend = _DummyCacheBackend()

    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.pipeline = None
    runner.cache_backend = None
    runner.offload_backend = None
    runner.od_config = SimpleNamespace(
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        cache_backend="cache_dit",
        cache_config={},
        model_class_name="NextStep11Pipeline",
        enforce_eager=True,
    )

    monkeypatch.setattr(model_runner_module, "LoadConfig", lambda: object())
    monkeypatch.setattr(model_runner_module, "DiffusersPipelineLoader", _DummyLoader)
    monkeypatch.setattr(model_runner_module, "DeviceMemoryProfiler", _DummyMemoryProfiler)
    monkeypatch.setattr(model_runner_module, "get_offload_backend", lambda od_config, device: None)
    monkeypatch.setattr(
        model_runner_module, "get_cache_backend", lambda cache_backend, cache_config: dummy_cache_backend
    )

    DiffusionModelRunner.load_model(runner)

    assert runner.cache_backend is None
    assert runner.od_config.cache_backend is None
    assert dummy_cache_backend.enabled is False


@pytest.mark.core_model
@pytest.mark.cpu
def test_set_forward_context_enters_vllm_config_contexts(monkeypatch):
    """Ensure `with set_forward_context(...):` enters vllm's context managers internally and calls desired vllm functions."""
    import vllm.config.vllm as vllm_config_module
    import vllm.ir
    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig

    from vllm_omni.diffusion.forward_context import (
        get_forward_context,
        is_forward_context_available,
        set_forward_context,
    )

    vllm_config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(),
    )
    calls = []

    @contextmanager
    def _set_current_vllm_config(cfg):
        calls.append(("set_current_vllm_config", cfg))
        yield
        calls.append(("set_current_vllm_config_exit", cfg))

    @contextmanager
    def _set_priority(*args, **kwargs):
        del args, kwargs
        calls.append(("ir_op_priority", None))
        yield
        calls.append(("ir_op_priority_exit", None))

    @contextmanager
    def _enable_torch_wrap(flag):
        calls.append(("enable_torch_wrap", flag))
        yield
        calls.append(("enable_torch_wrap_exit", flag))

    monkeypatch.setattr(vllm_config_module, "set_current_vllm_config", _set_current_vllm_config)
    monkeypatch.setattr(vllm_config.kernel_config.ir_op_priority, "set_priority", _set_priority)
    monkeypatch.setattr(vllm.ir, "enable_torch_wrap", _enable_torch_wrap)

    assert not is_forward_context_available()

    with set_forward_context(vllm_config=vllm_config):
        assert is_forward_context_available()
        assert get_forward_context().vllm_config is vllm_config

    assert not is_forward_context_available()
    assert calls == [
        ("set_current_vllm_config", vllm_config),
        ("ir_op_priority", None),
        ("enable_torch_wrap", vllm_config.compilation_config.ir_enable_torch_wrap),
        ("enable_torch_wrap_exit", vllm_config.compilation_config.ir_enable_torch_wrap),
        ("ir_op_priority_exit", None),
        ("set_current_vllm_config_exit", vllm_config),
    ]


@pytest.mark.core_model
@pytest.mark.cpu
def test_vllm_set_forward_context_implementation(monkeypatch):
    """Regression test: ensure that vLLM's set_forward_context implementation has changed."""
    import vllm.forward_context as vllm_forward_context
    import vllm.ir
    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig

    ERROR_MESSAGE = (
        "If this test fails, it likely means that vLLM's set_forward_context (vllm/forward_context.py) implementation has changed. "
        "In this case, we should update our forward_context (vllm_omni/diffusion/forward_context.py) as well. "
        "We should at least confirm that the `try: with (<what's inside?>): yield` part does not miss any information "
        "(typically by calling the same or similar stuff as vLLM). See #3352 for an example. "
        "Then, update this test to reflect the new implementation, and also update test_set_forward_context_enters_vllm_config_contexts."
    )

    vllm_config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(),
    )
    calls = []

    @contextmanager
    def _set_priority():
        calls.append(("ir_op_priority", None))
        yield
        calls.append(("ir_op_priority_exit", None))

    @contextmanager
    def _enable_torch_wrap(flag):
        calls.append(("enable_torch_wrap", flag))
        yield
        calls.append(("enable_torch_wrap_exit", flag))

    def _set_additional_forward_context(**kwargs):
        calls.append(("set_additional_forward_context", tuple(sorted(kwargs.keys()))))
        return {}

    monkeypatch.setattr(vllm_config.kernel_config.ir_op_priority, "set_priority", _set_priority)
    monkeypatch.setattr(vllm.ir, "enable_torch_wrap", _enable_torch_wrap)
    monkeypatch.setattr(
        vllm_forward_context.current_platform,
        "set_additional_forward_context",
        _set_additional_forward_context,
    )

    assert not vllm_forward_context.is_forward_context_available(), ERROR_MESSAGE

    with vllm_forward_context.set_forward_context(None, vllm_config):
        assert vllm_forward_context.is_forward_context_available(), ERROR_MESSAGE
        assert vllm_forward_context.get_forward_context().attn_metadata is None, ERROR_MESSAGE

    assert not vllm_forward_context.is_forward_context_available(), ERROR_MESSAGE
    assert calls == [
        (
            "set_additional_forward_context",
            (
                "attn_metadata",
                "batch_descriptor",
                "cudagraph_runtime_mode",
                "dp_metadata",
                "num_tokens",
                "num_tokens_across_dp",
                "ubatch_slices",
                "vllm_config",
            ),
        ),
        ("ir_op_priority", None),
        ("enable_torch_wrap", vllm_config.compilation_config.ir_enable_torch_wrap),
        ("enable_torch_wrap_exit", vllm_config.compilation_config.ir_enable_torch_wrap),
        ("ir_op_priority_exit", None),
    ], ERROR_MESSAGE
