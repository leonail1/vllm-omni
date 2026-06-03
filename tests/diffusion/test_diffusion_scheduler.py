# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import queue
import threading
import time
from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.diffusion.data import DiffusionOutput, DiffusionRequestAbortedError
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import (
    AdaptiveTokenSloStepScheduler,
    DiffusionRequestStatus,
    RequestScheduler,
    Scheduler,
    SchedulerInterface,
    SloStepScheduler,
    StepScheduler,
    TokenSloStepScheduler,
    TokenStepPreemptiveSloStepScheduler,
)
from vllm_omni.diffusion.sched.interface import CachedRequestData, NewRequestData
from vllm_omni.diffusion.worker.utils import RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_request(req_id: str) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[f"prompt_{req_id}"],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_ids=[req_id],
    )


def _make_request_output(req_id: str, *, error: str | None = None, finished: bool = True):
    return RunnerOutput(
        req_id=req_id,
        step_index=None,
        finished=finished,
        result=DiffusionOutput(output=None, error=error),
    )


def _make_step_output(
    req_id: str,
    step_index: int,
    *,
    finished: bool = False,
    error: str | None = None,
):
    return RunnerOutput(
        req_id=req_id,
        step_index=step_index,
        finished=finished,
        result=DiffusionOutput(output=None, error=error) if error is not None else None,
    )


def _make_step_request(
    req_id: str,
    *,
    num_inference_steps: int = 4,
    step_index: int | None = None,
    sampling_params: OmniDiffusionSamplingParams | None = None,
) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[f"prompt_{req_id}"],
        sampling_params=sampling_params
        or OmniDiffusionSamplingParams(
            num_inference_steps=num_inference_steps,
            step_index=step_index,
        ),
        request_ids=[req_id],
    )


def _make_slo_step_request(
    req_id: str,
    *,
    reference_cost_ms: float,
    slo_ms: float,
    num_inference_steps: int = 4,
    height: int | None = None,
    width: int | None = None,
) -> OmniDiffusionRequest:
    sampling_params = OmniDiffusionSamplingParams(
        height=height,
        width=width or height,
        num_inference_steps=num_inference_steps,
        reference_cost_ms=reference_cost_ms,
        slo_ms=slo_ms,
    )
    return _make_step_request(req_id, sampling_params=sampling_params)


def _new_ids(sched_output) -> list[str]:
    return [req.sched_req_id for req in sched_output.scheduled_new_reqs]


def _cached_ids(sched_output) -> list[str]:
    return list(sched_output.scheduled_cached_reqs.sched_req_ids)


class _StubScheduler(SchedulerInterface):
    def __init__(self, request: OmniDiffusionRequest, output) -> None:
        self._request = request
        self._output = output
        self.initialized_with = None
        self._sched_req_id = request.request_ids[0]
        self._state = None
        self._scheduled = False
        self.max_num_running_reqs = 1

    def initialize(self, od_config) -> None:
        self.initialized_with = od_config

    def add_request(self, request: OmniDiffusionRequest) -> str:
        assert request is self._request
        self._state = SimpleNamespace(sched_req_id=self._sched_req_id, req=request)
        return self._sched_req_id

    def schedule(self):
        if self._scheduled or self._state is None:
            return SimpleNamespace(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=CachedRequestData.make_empty(),
                scheduled_req_ids=[],
                is_empty=True,
            )
        self._scheduled = True
        return SimpleNamespace(
            scheduled_new_reqs=[NewRequestData.from_state(self._state)],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            scheduled_req_ids=[self._state.sched_req_id],
            is_empty=False,
        )

    def update_from_output(self, sched_output, output) -> set[str]:
        del sched_output
        assert output is self._output
        self._state.status = DiffusionRequestStatus.FINISHED_COMPLETED
        return {self._sched_req_id}

    def has_requests(self) -> bool:
        return not self._scheduled

    def get_request_state(self, sched_req_id: str):
        del sched_req_id
        return self._state

    def get_sched_req_id(self, request_id: str) -> str | None:
        if request_id in self._request.request_ids:
            return self._sched_req_id
        return None

    def pop_request_state(self, sched_req_id: str):
        del sched_req_id
        return self._state

    def preempt_request(self, sched_req_id: str) -> bool:
        del sched_req_id
        return False

    def finish_requests(self, sched_req_ids, status) -> None:
        del sched_req_ids, status
        return None

    def close(self) -> None:
        return None


class TestGetSamplingParamsKey:
    """Pure-function tests for the batch-compatibility key builder."""

    @staticmethod
    def _make(lora_int_id: int | None = None, lora_scale: float = 1.0) -> OmniDiffusionRequest:
        from vllm_omni.lora.request import LoRARequest

        sp = OmniDiffusionSamplingParams(num_inference_steps=2)
        if lora_int_id is not None:
            sp.lora_request = LoRARequest(
                lora_name=f"adapter-{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path=f"/tmp/lora-{lora_int_id}",
            )
        sp.lora_scale = lora_scale
        return OmniDiffusionRequest(
            prompts=["prompt"],
            sampling_params=sp,
            request_ids=[f"req-{lora_int_id}-{lora_scale}"],
        )

    def test_distinguishes_lora_id(self) -> None:
        from vllm_omni.diffusion.sched.base_scheduler import get_sampling_params_key

        assert get_sampling_params_key(self._make(lora_int_id=1)) != get_sampling_params_key(self._make(lora_int_id=2))

    def test_distinguishes_lora_scale(self) -> None:
        from vllm_omni.diffusion.sched.base_scheduler import get_sampling_params_key

        assert get_sampling_params_key(self._make(lora_int_id=1, lora_scale=0.5)) != get_sampling_params_key(
            self._make(lora_int_id=1, lora_scale=1.0)
        )

    def test_treats_no_lora_as_distinct_bucket(self) -> None:
        from vllm_omni.diffusion.sched.base_scheduler import get_sampling_params_key

        assert get_sampling_params_key(self._make(lora_int_id=None)) != get_sampling_params_key(
            self._make(lora_int_id=1)
        )

    def test_equal_for_same_lora_identity(self) -> None:
        from vllm_omni.diffusion.sched.base_scheduler import get_sampling_params_key

        a = get_sampling_params_key(self._make(lora_int_id=1, lora_scale=0.5))
        b = get_sampling_params_key(self._make(lora_int_id=1, lora_scale=0.5))
        assert a == b

    @staticmethod
    def _qwen_config() -> SimpleNamespace:
        return SimpleNamespace(
            step_execution=True,
            max_num_seqs=2,
            enforce_eager=True,
            cache_backend="none",
            diffusion_kv_cache_dtype=None,
            parallel_config=SimpleNamespace(
                sequence_parallel_size=1,
                ring_degree=1,
                cfg_parallel_size=1,
                use_hsdp=False,
            ),
            model_class_name="QwenImagePipeline",
            model="Qwen/Qwen-Image",
            additional_config={"diffusion_dynamic_step_batching_enabled": True},
        )

    @staticmethod
    def _make_qwen(
        *,
        request_id: str,
        height: int = 512,
        width: int = 512,
        negative_prompt: str | None = "low quality",
        true_cfg_scale: float = 4.0,
        lora_int_id: int | None = None,
    ) -> OmniDiffusionRequest:
        from vllm_omni.lora.request import LoRARequest

        sp = OmniDiffusionSamplingParams(
            num_inference_steps=2,
            height=height,
            width=width,
            true_cfg_scale=true_cfg_scale,
        )
        if lora_int_id is not None:
            sp.lora_request = LoRARequest(
                lora_name=f"adapter-{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path=f"/tmp/lora-{lora_int_id}",
            )
        prompt = {"prompt": f"prompt-{request_id}"}
        if negative_prompt is not None:
            prompt["negative_prompt"] = negative_prompt
        return OmniDiffusionRequest(
            prompts=[prompt],
            sampling_params=sp,
            request_id=request_id,
        )

    def test_qwen_dynamic_admission_ignores_resolution_and_cfg_scalar(self) -> None:
        from vllm_omni.diffusion.sched.base_scheduler import get_sampling_params_key

        config = self._qwen_config()
        a = get_sampling_params_key(
            self._make_qwen(request_id="a", height=512, width=512, true_cfg_scale=4.0),
            config,
        )
        b = get_sampling_params_key(
            self._make_qwen(request_id="b", height=1024, width=1024, true_cfg_scale=7.0),
            config,
        )
        assert a == b

    def test_qwen_dynamic_admission_requires_explicit_enable(self) -> None:
        from vllm_omni.diffusion.sched.base_scheduler import get_sampling_params_key

        config = self._qwen_config()
        config.additional_config = {}
        a = get_sampling_params_key(
            self._make_qwen(request_id="a", height=512, width=512, true_cfg_scale=4.0),
            config,
        )
        b = get_sampling_params_key(
            self._make_qwen(request_id="b", height=1024, width=1024, true_cfg_scale=7.0),
            config,
        )
        assert a != b

    def test_qwen_dynamic_admission_can_be_disabled(self) -> None:
        from vllm_omni.diffusion.sched.base_scheduler import get_sampling_params_key

        config = self._qwen_config()
        config.additional_config = {"diffusion_dynamic_step_batching_enabled": False}
        a = get_sampling_params_key(
            self._make_qwen(request_id="a", height=512, width=512, true_cfg_scale=4.0),
            config,
        )
        b = get_sampling_params_key(
            self._make_qwen(request_id="b", height=1024, width=1024, true_cfg_scale=7.0),
            config,
        )
        assert a != b

    def test_qwen_dynamic_admission_keeps_cfg_shape_and_lora(self) -> None:
        from vllm_omni.diffusion.sched.base_scheduler import get_sampling_params_key

        config = self._qwen_config()
        cfg = get_sampling_params_key(self._make_qwen(request_id="cfg"), config)
        no_cfg = get_sampling_params_key(
            self._make_qwen(request_id="no-cfg", negative_prompt=None),
            config,
        )
        scale_one = get_sampling_params_key(
            self._make_qwen(request_id="scale-one", true_cfg_scale=1.0),
            config,
        )
        lora_1 = get_sampling_params_key(self._make_qwen(request_id="lora-1", lora_int_id=1), config)
        lora_2 = get_sampling_params_key(self._make_qwen(request_id="lora-2", lora_int_id=2), config)

        assert cfg != no_cfg
        assert cfg != scale_one
        assert lora_1 != lora_2


class TestRequestScheduler:
    def setup_method(self) -> None:
        self.scheduler: RequestScheduler = RequestScheduler()
        self.scheduler.initialize(SimpleNamespace())

    def test_single_request_success_lifecycle(self) -> None:
        req_id = self.scheduler.add_request(_make_request("a"))
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.WAITING

        sched_output = self.scheduler.schedule()
        assert _new_ids(sched_output) == [req_id]
        assert _cached_ids(sched_output) == []
        assert sched_output.num_running_reqs == 1
        assert sched_output.num_waiting_reqs == 0

        finished = self.scheduler.update_from_output(sched_output, _make_request_output(req_id))
        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        assert self.scheduler.has_requests() is False

    def test_error_output_marks_finished_error(self) -> None:
        req_id = self.scheduler.add_request(_make_request("err"))

        sched_output = self.scheduler.schedule()
        finished = self.scheduler.update_from_output(
            sched_output,
            _make_request_output(req_id, error="worker failed"),
        )

        assert finished == {req_id}
        state = self.scheduler.get_request_state(req_id)
        assert state.status == DiffusionRequestStatus.FINISHED_ERROR
        assert state.error == "worker failed"

    def test_empty_output_without_error_marks_completed(self) -> None:
        req_id = self.scheduler.add_request(_make_request("empty"))

        sched_output = self.scheduler.schedule()
        finished = self.scheduler.update_from_output(sched_output, _make_request_output(req_id))

        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED

    def test_fifo_single_request_scheduling(self) -> None:
        req_id_a = self.scheduler.add_request(_make_request("a"))
        req_id_b = self.scheduler.add_request(_make_request("b"))

        first = self.scheduler.schedule()
        assert _new_ids(first) == [req_id_a]
        assert _cached_ids(first) == []
        assert first.num_running_reqs == 1
        assert first.num_waiting_reqs == 1

        # Request A is still running; scheduling again should not pull B.
        second = self.scheduler.schedule()
        assert _new_ids(second) == []
        assert _cached_ids(second) == [req_id_a]
        assert second.num_running_reqs == 1
        assert second.num_waiting_reqs == 1

        self.scheduler.update_from_output(first, _make_request_output(req_id_a))

        third = self.scheduler.schedule()
        assert _new_ids(third) == [req_id_b]
        assert _cached_ids(third) == []
        assert third.num_running_reqs == 1
        assert third.num_waiting_reqs == 0

    def test_batches_compatible_requests_up_to_max_num_seqs(self) -> None:
        scheduler = RequestScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=2))

        req_id_a = scheduler.add_request(_make_request("a"))
        req_id_b = scheduler.add_request(_make_request("b"))

        sched_output = scheduler.schedule()

        assert _new_ids(sched_output) == [req_id_a, req_id_b]
        assert sched_output.num_running_reqs == 2
        assert sched_output.num_waiting_reqs == 0

    def test_qwen_dynamic_batches_different_resolution_and_cfg_scale(self) -> None:
        scheduler = RequestScheduler()
        scheduler.initialize(TestGetSamplingParamsKey._qwen_config())

        req_a = scheduler.add_request(
            TestGetSamplingParamsKey._make_qwen(
                request_id="a",
                height=512,
                width=512,
                true_cfg_scale=4.0,
            )
        )
        req_b = scheduler.add_request(
            TestGetSamplingParamsKey._make_qwen(
                request_id="b",
                height=1024,
                width=1024,
                true_cfg_scale=7.0,
            )
        )

        sched_output = scheduler.schedule()

        assert _new_ids(sched_output) == [req_a, req_b]
        assert sched_output.num_running_reqs == 2
        assert sched_output.num_waiting_reqs == 0

    def test_qwen_dynamic_does_not_mix_cfg_disabled_scale_one(self) -> None:
        scheduler = RequestScheduler()
        scheduler.initialize(
            SimpleNamespace(
                step_execution=True,
                max_num_seqs=3,
                model_class_name="QwenImagePipeline",
                model="Qwen/Qwen-Image",
            )
        )

        req_a = scheduler.add_request(TestGetSamplingParamsKey._make_qwen(request_id="a", true_cfg_scale=4.0))
        scheduler.add_request(TestGetSamplingParamsKey._make_qwen(request_id="b", true_cfg_scale=1.0))
        scheduler.add_request(TestGetSamplingParamsKey._make_qwen(request_id="c", true_cfg_scale=7.0))

        sched_output = scheduler.schedule()

        assert _new_ids(sched_output) == [req_a]
        assert sched_output.num_running_reqs == 1
        assert sched_output.num_waiting_reqs == 2

    def test_incompatible_waiting_head_blocks_later_compatible_request(self) -> None:
        scheduler = RequestScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=3))

        req_id_a = scheduler.add_request(_make_request("a"))
        req_id_b = scheduler.add_request(
            OmniDiffusionRequest(
                prompts=["prompt_b"],
                sampling_params=OmniDiffusionSamplingParams(do_classifier_free_guidance=True),
                request_ids=["b"],
            )
        )
        scheduler.add_request(_make_request("c"))

        first = scheduler.schedule()

        assert _new_ids(first) == [req_id_a]
        assert first.num_running_reqs == 1
        assert first.num_waiting_reqs == 2

        scheduler.update_from_output(first, _make_request_output(req_id_a))
        second = scheduler.schedule()

        assert _new_ids(second) == [req_id_b]
        assert second.num_running_reqs == 1
        assert second.num_waiting_reqs == 1

    def test_abort_request_for_waiting_and_running(self) -> None:
        req_id_a = self.scheduler.add_request(_make_request("a"))
        req_id_b = self.scheduler.add_request(_make_request("b"))

        # Abort waiting request.
        self.scheduler.finish_requests(req_id_b, DiffusionRequestStatus.FINISHED_ABORTED)
        state_b = self.scheduler.get_request_state(req_id_b)
        assert state_b.status == DiffusionRequestStatus.FINISHED_ABORTED

        first = self.scheduler.schedule()
        assert first.finished_req_ids == {req_id_b}
        # A should still run normally.
        assert _new_ids(first) == [req_id_a]

        # B is already marked finished aborted, scheduling again should not pull it.
        second = self.scheduler.schedule()
        assert second.finished_req_ids == set()

        # Abort running request.
        self.scheduler.finish_requests(req_id_a, DiffusionRequestStatus.FINISHED_ABORTED)
        state_a = self.scheduler.get_request_state(req_id_a)
        assert state_a.status == DiffusionRequestStatus.FINISHED_ABORTED

        assert self.scheduler.has_requests() is False
        assert self.scheduler.schedule().scheduled_req_ids == []

    def test_has_requests_state_transition(self) -> None:
        assert self.scheduler.has_requests() is False

        req_id = self.scheduler.add_request(_make_request("has"))
        assert self.scheduler.has_requests() is True

        sched_output = self.scheduler.schedule()
        assert self.scheduler.has_requests() is True

        self.scheduler.update_from_output(sched_output, _make_request_output(req_id))
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        assert self.scheduler.has_requests() is False

    def test_request_id_mapping_lifecycle(self) -> None:
        request = OmniDiffusionRequest(
            prompts=["prompt_map_a", "prompt_map_b"],
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
            request_ids=["map-a", "map-b"],
            request_id="map-parent",
        )

        sched_req_id = self.scheduler.add_request(request)

        assert self.scheduler.get_sched_req_id("map-a") == sched_req_id
        assert self.scheduler.get_sched_req_id("map-b") == sched_req_id
        assert self.scheduler.get_sched_req_id("map-parent") == sched_req_id

        self.scheduler.pop_request_state(sched_req_id)

        assert self.scheduler.get_sched_req_id("map-a") is None
        assert self.scheduler.get_sched_req_id("map-b") is None
        assert self.scheduler.get_sched_req_id("map-parent") is None

    def test_parent_request_id_registration_failure_rolls_back_child_ids(self) -> None:
        self.scheduler.add_request(
            OmniDiffusionRequest(
                prompts=["prompt_existing"],
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
                request_ids=["existing-child"],
                request_id="duplicate-parent",
            )
        )

        colliding_request = OmniDiffusionRequest(
            prompts=["prompt_unique"],
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
            request_ids=["unique-child"],
            request_id="duplicate-parent",
        )

        with pytest.raises(ValueError, match="duplicate-parent"):
            self.scheduler.add_request(colliding_request)

        assert self.scheduler.get_request_state("unique-child") is None
        assert self.scheduler.get_sched_req_id("unique-child") is None

        valid_request = OmniDiffusionRequest(
            prompts=["prompt_unique"],
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
            request_ids=["unique-child"],
            request_id="unique-parent",
        )
        sched_req_id = self.scheduler.add_request(valid_request)

        assert sched_req_id == "unique-child"
        assert self.scheduler.get_sched_req_id("unique-child") == sched_req_id
        assert self.scheduler.get_sched_req_id("unique-parent") == sched_req_id


class TestDiffusionEngine:
    @staticmethod
    def _qwen_dynamic_slo_config(dynamic_enabled: bool = True) -> SimpleNamespace:
        return SimpleNamespace(
            step_execution=True,
            model_class_name="QwenImagePipeline",
            enforce_eager=True,
            cache_backend="none",
            diffusion_kv_cache_dtype=None,
            parallel_config=SimpleNamespace(
                sequence_parallel_size=1,
                ring_degree=1,
                cfg_parallel_size=1,
                use_hsdp=False,
            ),
            additional_config={
                "diffusion_scheduler_policy": "slo",
                "diffusion_dynamic_step_batching_enabled": dynamic_enabled,
            },
        )

    def test_rejects_qwen_dynamic_batching_with_shape_bucket_slo_scheduler(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.step_execution = True

        with pytest.raises(ValueError, match="dynamic step batching cannot be combined"):
            engine._make_default_scheduler(self._qwen_dynamic_slo_config(dynamic_enabled=True))

    def test_allows_shape_bucket_slo_when_qwen_dynamic_batching_disabled(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.step_execution = True

        scheduler = engine._make_default_scheduler(self._qwen_dynamic_slo_config(dynamic_enabled=False))

        assert isinstance(scheduler, SloStepScheduler)

    def test_allows_qwen_dynamic_batching_with_token_slo_scheduler(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.step_execution = True
        config = self._qwen_dynamic_slo_config(dynamic_enabled=True)
        config.additional_config["diffusion_scheduler_policy"] = "slo_no_preemption_token_guarded"

        scheduler = engine._make_default_scheduler(config)

        assert isinstance(scheduler, TokenSloStepScheduler)

    def test_allows_qwen_dynamic_batching_with_adaptive_token_slo_scheduler(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.step_execution = True
        config = self._qwen_dynamic_slo_config(dynamic_enabled=True)
        config.additional_config["diffusion_scheduler_policy"] = "slo_no_preemption_token_adaptive"

        scheduler = engine._make_default_scheduler(config)

        assert isinstance(scheduler, AdaptiveTokenSloStepScheduler)

    @pytest.mark.parametrize(
        "policy",
        [
            "slo_no_preemption_token_objective",
            "slo_token_stagepool_objective",
        ],
    )
    def test_allows_qwen_dynamic_batching_with_token_objective_scheduler(self, policy: str) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.step_execution = True
        config = self._qwen_dynamic_slo_config(dynamic_enabled=True)
        config.additional_config["diffusion_scheduler_policy"] = policy

        scheduler = engine._make_default_scheduler(config)

        assert isinstance(scheduler, TokenSloStepScheduler)

    @pytest.mark.parametrize(
        "policy",
        [
            "slo_token_step_preemptive",
            "slo_step_preemptive_token",
            "slo_token_preemptive",
        ],
    )
    def test_allows_qwen_dynamic_batching_with_token_step_preemptive_scheduler(self, policy: str) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.step_execution = True
        config = self._qwen_dynamic_slo_config(dynamic_enabled=True)
        config.additional_config["diffusion_scheduler_policy"] = policy

        scheduler = engine._make_default_scheduler(config)

        assert isinstance(scheduler, TokenStepPreemptiveSloStepScheduler)

    def test_add_req_and_wait_for_response_single_path(self, mocker: MockerFixture) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.scheduler = RequestScheduler()
        engine.scheduler.initialize(SimpleNamespace())
        engine._rpc_lock = threading.RLock()
        engine._cv = threading.Condition(engine._rpc_lock)
        engine._closed = False
        engine.abort_queue = queue.Queue()

        request = _make_request("engine")
        runner_output = _make_request_output("engine")
        engine.execute_fn = mocker.Mock(return_value=runner_output)

        output = engine.add_req_and_wait_for_response(request)

        assert output is runner_output.result
        engine.execute_fn.assert_called_once()

    def test_supports_scheduler_interface_injection(self, mocker: MockerFixture) -> None:
        request = _make_request("engine_iface")
        runner_output = _make_request_output("engine_iface")
        scheduler = _StubScheduler(request, runner_output)

        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.scheduler = scheduler
        engine._rpc_lock = threading.RLock()
        engine._cv = threading.Condition(engine._rpc_lock)
        engine._closed = False
        engine.abort_queue = queue.Queue()
        engine.execute_fn = mocker.Mock(return_value=runner_output)

        output = engine.add_req_and_wait_for_response(request)

        assert output is runner_output.result
        engine.execute_fn.assert_called_once()

    def test_initializes_injected_scheduler(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mocker: MockerFixture,
    ) -> None:
        request = _make_request("init")
        scheduler = _StubScheduler(request, DiffusionOutput(output=None))
        od_config = SimpleNamespace(model_class_name="mock_model")
        fake_executor_cls = mocker.Mock(return_value=mocker.Mock())

        monkeypatch.setattr(
            "vllm_omni.diffusion.diffusion_engine.get_diffusion_post_process_func",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.diffusion_engine.get_diffusion_pre_process_func",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.diffusion_engine.DiffusionExecutor.get_class",
            lambda *args, **kwargs: fake_executor_cls,
        )
        monkeypatch.setattr(DiffusionEngine, "_dummy_run", lambda self: None)

        DiffusionEngine(od_config, scheduler=scheduler)

        assert scheduler.initialized_with is od_config
        fake_executor_cls.assert_called_once_with(od_config)

    def test_scheduler_alias_keeps_default_request_scheduler(self) -> None:
        scheduler = Scheduler()
        scheduler.initialize(SimpleNamespace())

        req_id = scheduler.add_request(_make_request("alias"))
        sched_output = scheduler.schedule()
        finished = scheduler.update_from_output(sched_output, _make_request_output(req_id))

        assert req_id in finished
        assert scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED

    @pytest.mark.asyncio
    async def test_step_raises_aborted_error(self, mocker: MockerFixture) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine._closed = False
        engine._loop_started = True
        engine._init_lock = asyncio.Lock()
        engine.main_loop = asyncio.get_running_loop()
        engine.stop_event = threading.Event()
        engine.pre_process_func = None
        engine.async_add_req_and_wait_for_response = mocker.AsyncMock(
            return_value=DiffusionOutput(aborted=True, abort_message="Request req-abort aborted.")
        )

        with pytest.raises(DiffusionRequestAbortedError, match="Request req-abort aborted"):
            await engine.step(_make_request("req-abort"))

    def test_abort_queue_marks_request_finished_aborted(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine._rpc_lock = threading.RLock()
        engine._cv = threading.Condition(engine._rpc_lock)
        engine._closed = False
        engine.scheduler = RequestScheduler()
        engine.scheduler.initialize(SimpleNamespace())
        engine.abort_queue = queue.Queue()

        req_id = engine.scheduler.add_request(_make_request("req-abort"))
        engine.abort("req-abort")
        engine._process_aborts_queue()

        assert engine.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_ABORTED

    def test_finalize_finished_request_returns_aborted_output(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.scheduler = RequestScheduler()
        engine.scheduler.initialize(SimpleNamespace())

        req_id = engine.scheduler.add_request(_make_request("req-finalize"))
        engine.scheduler.finish_requests(req_id, DiffusionRequestStatus.FINISHED_ABORTED)

        output = engine._finalize_finished_request(req_id)

        assert output.aborted is True
        assert output.abort_message == "Request req-finalize aborted."

    def test_initializes_step_scheduler_when_step_execution_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mocker: MockerFixture,
    ) -> None:
        od_config = SimpleNamespace(model_class_name="mock_model")
        od_config.step_execution = True
        fake_executor = mocker.Mock()
        fake_executor_cls = mocker.Mock(return_value=fake_executor)

        monkeypatch.setattr(
            "vllm_omni.diffusion.diffusion_engine.get_diffusion_post_process_func",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.diffusion_engine.get_diffusion_pre_process_func",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.diffusion_engine.DiffusionExecutor.get_class",
            lambda *args, **kwargs: fake_executor_cls,
        )
        monkeypatch.setattr(DiffusionEngine, "_dummy_run", lambda self: None)
        engine = DiffusionEngine(od_config)

        assert isinstance(engine.scheduler, StepScheduler)
        assert engine.execute_fn is fake_executor.execute_step
        fake_executor_cls.assert_called_once_with(od_config)

    def test_dummy_run_raises_on_output_error(self, mocker: MockerFixture) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.od_config = SimpleNamespace(model_class_name="mock_model", diffusion_load_format="default")
        engine.pre_process_func = None
        engine.add_req_and_wait_for_response = mocker.Mock(return_value=DiffusionOutput(error="boom"))

        with pytest.raises(RuntimeError, match="Dummy run failed: boom"):
            engine._dummy_run()


class TestStepScheduler:
    def setup_method(self) -> None:
        self.scheduler: StepScheduler = StepScheduler()
        self.scheduler.initialize(SimpleNamespace())

    def test_single_request_step_lifecycle(self) -> None:
        request = _make_step_request("step", num_inference_steps=3)
        req_id = self.scheduler.add_request(request)

        first = self.scheduler.schedule()
        assert _new_ids(first) == [req_id]
        assert _cached_ids(first) == []
        assert first.num_running_reqs == 1
        assert first.num_waiting_reqs == 0

        finished = self.scheduler.update_from_output(first, _make_step_output(req_id, step_index=1))
        assert finished == set()
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.RUNNING
        assert request.sampling_params.step_index == 1
        assert self.scheduler.has_requests() is True

        second = self.scheduler.schedule()
        assert _new_ids(second) == []
        assert _cached_ids(second) == [req_id]
        assert second.num_running_reqs == 1
        assert second.num_waiting_reqs == 0

        finished = self.scheduler.update_from_output(second, _make_step_output(req_id, step_index=2))
        assert finished == set()
        assert request.sampling_params.step_index == 2

        third = self.scheduler.schedule()
        assert _new_ids(third) == []
        assert _cached_ids(third) == [req_id]

        finished = self.scheduler.update_from_output(
            third,
            _make_step_output(req_id, step_index=3, finished=True),
        )
        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        assert request.sampling_params.step_index == 3
        assert self.scheduler.has_requests() is False

    def test_fifo_single_request_scheduling(self) -> None:
        req_id_a = self.scheduler.add_request(_make_step_request("a", num_inference_steps=2))
        req_id_b = self.scheduler.add_request(_make_step_request("b", num_inference_steps=2))

        first = self.scheduler.schedule()
        assert _new_ids(first) == [req_id_a]
        assert _cached_ids(first) == []
        assert first.num_running_reqs == 1
        assert first.num_waiting_reqs == 1

        finished = self.scheduler.update_from_output(first, _make_step_output(req_id_a, step_index=1))
        assert finished == set()

        second = self.scheduler.schedule()
        assert _new_ids(second) == []
        assert _cached_ids(second) == [req_id_a]
        assert second.num_running_reqs == 1
        assert second.num_waiting_reqs == 1

        finished = self.scheduler.update_from_output(
            second,
            _make_step_output(req_id_a, step_index=2, finished=True),
        )
        assert finished == {req_id_a}

        third = self.scheduler.schedule()
        assert _new_ids(third) == [req_id_b]
        assert _cached_ids(third) == []
        assert third.num_running_reqs == 1
        assert third.num_waiting_reqs == 0

    def test_error_output_marks_finished_error(self) -> None:
        req_id = self.scheduler.add_request(_make_step_request("err", num_inference_steps=3))

        sched_output = self.scheduler.schedule()
        assert _new_ids(sched_output) == [req_id]
        finished = self.scheduler.update_from_output(
            sched_output,
            _make_step_output(req_id, step_index=1, finished=True, error="worker failed"),
        )

        assert finished == {req_id}
        state = self.scheduler.get_request_state(req_id)
        assert state.status == DiffusionRequestStatus.FINISHED_ERROR
        assert state.error == "worker failed"
        assert self.scheduler.has_requests() is False

    def test_missing_step_index_marks_finished_error(self) -> None:
        req_id = self.scheduler.add_request(_make_step_request("missing", num_inference_steps=3))

        sched_output = self.scheduler.schedule()
        finished = self.scheduler.update_from_output(
            sched_output,
            RunnerOutput(
                req_id=req_id,
                step_index=None,
                finished=True,
                result=None,
            ),
        )

        assert finished == {req_id}
        state = self.scheduler.get_request_state(req_id)
        assert state.status == DiffusionRequestStatus.FINISHED_ERROR
        assert state.error == "Missing step_index in RunnerOutput"

    def test_abort_request_for_waiting_and_running(self) -> None:
        req_id_a = self.scheduler.add_request(_make_step_request("a", num_inference_steps=2))
        req_id_b = self.scheduler.add_request(_make_step_request("b", num_inference_steps=2))

        self.scheduler.finish_requests(req_id_b, DiffusionRequestStatus.FINISHED_ABORTED)
        assert self.scheduler.get_request_state(req_id_b).status == DiffusionRequestStatus.FINISHED_ABORTED

        running = self.scheduler.schedule()
        assert _new_ids(running) == [req_id_a]

        self.scheduler.finish_requests(req_id_a, DiffusionRequestStatus.FINISHED_ABORTED)
        assert self.scheduler.get_request_state(req_id_a).status == DiffusionRequestStatus.FINISHED_ABORTED
        assert self.scheduler.has_requests() is False

    def test_has_requests_state_transition(self) -> None:
        assert self.scheduler.has_requests() is False

        req_id = self.scheduler.add_request(_make_step_request("has", num_inference_steps=2))
        assert self.scheduler.has_requests() is True

        sched_output = self.scheduler.schedule()
        assert self.scheduler.has_requests() is True

        finished = self.scheduler.update_from_output(
            sched_output,
            _make_step_output(req_id, step_index=2, finished=True),
        )
        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        assert self.scheduler.has_requests() is False

    def test_scheduled_request_aborted_before_update_is_returned_finished(self) -> None:
        req_id = self.scheduler.add_request(_make_step_request("abort-late", num_inference_steps=2))

        sched_output = self.scheduler.schedule()
        self.scheduler.finish_requests(req_id, DiffusionRequestStatus.FINISHED_ABORTED)

        finished = self.scheduler.update_from_output(
            sched_output,
            _make_step_output(req_id, step_index=1),
        )
        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_ABORTED

    def test_batches_compatible_step_requests(self) -> None:
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=2))

        req_a = scheduler.add_request(_make_step_request("a"))
        req_b = scheduler.add_request(_make_step_request("b"))

        sched_output = scheduler.schedule()

        assert _new_ids(sched_output) == [req_a, req_b]
        assert sched_output.num_running_reqs == 2
        assert sched_output.num_waiting_reqs == 0

    def test_step_batch_allows_different_num_inference_steps(self) -> None:
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=2))

        req_a = scheduler.add_request(_make_step_request("a", num_inference_steps=2))
        req_b = scheduler.add_request(_make_step_request("b", num_inference_steps=4))

        sched_output = scheduler.schedule()

        assert _new_ids(sched_output) == [req_a, req_b]
        assert sched_output.num_running_reqs == 2
        assert sched_output.num_waiting_reqs == 0

    def test_step_batch_rejects_different_sampling_key(self) -> None:
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=3))

        req_a = scheduler.add_request(_make_step_request("a"))
        req_b = scheduler.add_request(
            _make_step_request(
                "b",
                sampling_params=OmniDiffusionSamplingParams(
                    do_classifier_free_guidance=True,
                    num_inference_steps=4,
                ),
            )
        )
        scheduler.add_request(_make_step_request("c"))

        sched_output = scheduler.schedule()

        assert _new_ids(sched_output) == [req_a]
        assert sched_output.num_running_reqs == 1
        assert sched_output.num_waiting_reqs == 2

        scheduler.update_from_output(
            sched_output,
            _make_step_output(req_a, step_index=4, finished=True),
        )
        second = scheduler.schedule()

        assert _new_ids(second) == [req_b]
        assert second.num_running_reqs == 1
        assert second.num_waiting_reqs == 1

    def test_step_batch_co_schedules_requests_sharing_lora(self) -> None:
        """Multiple requests with the same LoRA (id + scale) co-batch."""
        from vllm_omni.lora.request import LoRARequest

        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=3))

        lora = LoRARequest(lora_name="adapter", lora_int_id=42, lora_path="/tmp/lora")

        def _with_lora(req_id: str) -> OmniDiffusionRequest:
            sp = OmniDiffusionSamplingParams(num_inference_steps=4)
            sp.lora_request = lora
            sp.lora_scale = 0.5
            return _make_step_request(req_id, sampling_params=sp)

        req_a = scheduler.add_request(_with_lora("a"))
        req_b = scheduler.add_request(_with_lora("b"))
        req_c = scheduler.add_request(_with_lora("c"))

        sched_output = scheduler.schedule()

        assert _new_ids(sched_output) == [req_a, req_b, req_c]
        assert sched_output.num_running_reqs == 3
        assert sched_output.num_waiting_reqs == 0

    def test_step_batch_separates_requests_with_different_lora_ids(self) -> None:
        """Different LoRA adapters → distinct batches admitted in FIFO order."""
        from vllm_omni.lora.request import LoRARequest

        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=4))

        lora_a = LoRARequest(lora_name="adapter-A", lora_int_id=1, lora_path="/tmp/lora-a")
        lora_b = LoRARequest(lora_name="adapter-B", lora_int_id=2, lora_path="/tmp/lora-b")

        def _build(req_id: str, lora: LoRARequest) -> OmniDiffusionRequest:
            sp = OmniDiffusionSamplingParams(num_inference_steps=2)
            sp.lora_request = lora
            return _make_step_request(req_id, sampling_params=sp)

        req_a1 = scheduler.add_request(_build("a1", lora_a))
        req_b1 = scheduler.add_request(_build("b1", lora_b))
        req_a2 = scheduler.add_request(_build("a2", lora_a))

        # Strict FIFO admission: a1 starts; b1 (different LoRA) blocks the
        # queue head, so a2 (compatible with a1) is *not* skipped ahead.
        first = scheduler.schedule()
        assert _new_ids(first) == [req_a1]
        assert first.num_waiting_reqs == 2

        # Drain a1 → b1 becomes head-of-line and is admitted with its LoRA.
        scheduler.update_from_output(first, _make_step_output(req_a1, step_index=2, finished=True))
        second = scheduler.schedule()
        assert _new_ids(second) == [req_b1]
        assert second.num_waiting_reqs == 1

        # Drain b1 → a2 is admitted next; LoRA-A is re-activated for it.
        scheduler.update_from_output(second, _make_step_output(req_b1, step_index=2, finished=True))
        third = scheduler.schedule()
        assert _new_ids(third) == [req_a2]
        assert third.num_waiting_reqs == 0

    def test_step_batch_separates_requests_with_different_lora_scale(self) -> None:
        """Same adapter id but different scales → still separate batches."""
        from vllm_omni.lora.request import LoRARequest

        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=4))

        lora = LoRARequest(lora_name="adapter", lora_int_id=7, lora_path="/tmp/lora")

        def _build(req_id: str, scale: float) -> OmniDiffusionRequest:
            sp = OmniDiffusionSamplingParams(num_inference_steps=2)
            sp.lora_request = lora
            sp.lora_scale = scale
            return _make_step_request(req_id, sampling_params=sp)

        req_full = scheduler.add_request(_build("full", 1.0))
        req_half = scheduler.add_request(_build("half", 0.5))

        sched_output = scheduler.schedule()

        admitted = _new_ids(sched_output)
        assert admitted == [req_full]
        assert req_half not in admitted
        assert sched_output.num_waiting_reqs == 1

    def test_step_batch_separates_lora_from_no_lora(self) -> None:
        """A LoRA request and a no-LoRA request do not share a batch."""
        from vllm_omni.lora.request import LoRARequest

        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=4))

        lora = LoRARequest(lora_name="adapter", lora_int_id=3, lora_path="/tmp/lora")

        sp_with = OmniDiffusionSamplingParams(num_inference_steps=2)
        sp_with.lora_request = lora
        req_with = scheduler.add_request(_make_step_request("with", sampling_params=sp_with))
        req_without = scheduler.add_request(_make_step_request("without", num_inference_steps=2))

        sched_output = scheduler.schedule()

        admitted = _new_ids(sched_output)
        assert admitted == [req_with]
        assert req_without not in admitted
        assert sched_output.num_waiting_reqs == 1

    def test_preempt_request_preserves_step_index(self) -> None:
        request = _make_step_request("preempt", num_inference_steps=3)
        req_id = self.scheduler.add_request(request)

        first = self.scheduler.schedule()
        assert self.scheduler.update_from_output(first, _make_step_output(req_id, step_index=1)) == set()
        assert request.sampling_params.step_index == 1

        second = self.scheduler.schedule()
        assert _cached_ids(second) == [req_id]
        assert self.scheduler.preempt_request(req_id) is True
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.PREEMPTED
        assert request.sampling_params.step_index == 1

        third = self.scheduler.schedule()
        assert _cached_ids(third) == [req_id]
        assert request.sampling_params.step_index == 1

    @pytest.mark.parametrize(
        ("sampling_params", "expected_steps"),
        [
            (
                OmniDiffusionSamplingParams(
                    timesteps=torch.tensor([1.0, 0.5, 0.0]),
                    sigmas=[1.0, 0.5, 0.25, 0.0],
                    num_inference_steps=5,
                ),
                3,
            ),
            (
                OmniDiffusionSamplingParams(
                    sigmas=[1.0, 0.5],
                    num_inference_steps=5,
                ),
                2,
            ),
            (
                OmniDiffusionSamplingParams(
                    num_inference_steps=4,
                ),
                4,
            ),
        ],
    )
    def test_total_steps_priority(self, sampling_params: OmniDiffusionSamplingParams, expected_steps: int) -> None:
        request = _make_step_request("priority", sampling_params=sampling_params)
        req_id = self.scheduler.add_request(request)

        for _ in range(expected_steps - 1):
            sched_output = self.scheduler.schedule()
            assert sched_output.scheduled_req_ids == [req_id]
            next_step = request.sampling_params.step_index + 1
            assert (
                self.scheduler.update_from_output(
                    sched_output,
                    _make_step_output(req_id, step_index=next_step),
                )
                == set()
            )

        final_output = self.scheduler.schedule()
        assert final_output.scheduled_req_ids == [req_id]
        assert self.scheduler.update_from_output(
            final_output,
            _make_step_output(req_id, step_index=expected_steps, finished=True),
        ) == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED

    @pytest.mark.parametrize(
        "sampling_params",
        [
            OmniDiffusionSamplingParams(num_inference_steps=0),
            OmniDiffusionSamplingParams(num_inference_steps=3, step_index=3),
            OmniDiffusionSamplingParams(num_inference_steps=3, step_index=-1),
        ],
    )
    def test_rejects_invalid_initial_step_state(self, sampling_params: OmniDiffusionSamplingParams) -> None:
        request = _make_step_request("invalid", sampling_params=sampling_params)

        with pytest.raises(ValueError):
            self.scheduler.add_request(request)


class TestSloStepScheduler:
    def _make_scheduler(
        self,
        max_num_seqs: int = 1,
        slo_config: dict | None = None,
        *,
        model: str = "Qwen/Qwen-Image",
    ) -> SloStepScheduler:
        config = {"batch_growth_alpha": 0.0}
        if slo_config:
            config.update(slo_config)
        scheduler = SloStepScheduler()
        scheduler.initialize(
            SimpleNamespace(
                model=model,
                max_num_seqs=max_num_seqs,
                additional_config={"diffusion_slo_scheduler": config},
            )
        )
        return scheduler

    def test_skips_incompatible_fifo_head_to_form_urgent_bucket(self) -> None:
        scheduler = self._make_scheduler(max_num_seqs=3)

        req_a = scheduler.add_request(_make_slo_step_request("a", reference_cost_ms=300, slo_ms=1000))
        scheduler.add_request(_make_slo_step_request("b", reference_cost_ms=100, slo_ms=10000, height=768))
        req_c = scheduler.add_request(_make_slo_step_request("c", reference_cost_ms=300, slo_ms=1000))
        req_d = scheduler.add_request(_make_slo_step_request("d", reference_cost_ms=300, slo_ms=1000))

        sched_output = scheduler.schedule()

        assert _new_ids(sched_output) == [req_a, req_c, req_d]
        assert sched_output.num_running_reqs == 3
        assert sched_output.num_waiting_reqs == 1

    def test_uses_completion_aware_laxity_instead_of_earliest_deadline_only(self) -> None:
        scheduler = self._make_scheduler(max_num_seqs=1)

        scheduler.add_request(_make_slo_step_request("early-cheap", reference_cost_ms=100, slo_ms=2000))
        later_expensive = scheduler.add_request(
            _make_slo_step_request("later-expensive", reference_cost_ms=2900, slo_ms=3000, height=768)
        )

        sched_output = scheduler.schedule()

        assert _new_ids(sched_output) == [later_expensive]
        assert sched_output.num_waiting_reqs == 1

    def test_can_skip_running_bucket_at_step_boundary(self) -> None:
        scheduler = self._make_scheduler(max_num_seqs=2)

        running = scheduler.add_request(_make_slo_step_request("running", reference_cost_ms=100, slo_ms=10000))
        first = scheduler.schedule()
        assert _new_ids(first) == [running]
        assert scheduler.update_from_output(first, _make_step_output(running, step_index=1)) == set()

        urgent = scheduler.add_request(_make_slo_step_request("urgent", reference_cost_ms=100, slo_ms=200, height=768))
        second = scheduler.schedule()

        assert _new_ids(second) == [urgent]
        assert _cached_ids(second) == []
        assert scheduler.get_request_state(running).status == DiffusionRequestStatus.PREEMPTED
        assert running in list(scheduler._waiting)

    def test_does_not_admit_new_bucket_when_resident_slots_are_full(self) -> None:
        scheduler = self._make_scheduler(max_num_seqs=1)

        running = scheduler.add_request(_make_slo_step_request("running", reference_cost_ms=100, slo_ms=10000))
        first = scheduler.schedule()
        assert _new_ids(first) == [running]
        assert scheduler.update_from_output(first, _make_step_output(running, step_index=1)) == set()

        urgent = scheduler.add_request(_make_slo_step_request("urgent", reference_cost_ms=100, slo_ms=200, height=768))
        second = scheduler.schedule()

        assert _new_ids(second) == []
        assert _cached_ids(second) == [running]
        assert scheduler.get_request_state(urgent).status == DiffusionRequestStatus.WAITING
        assert scheduler.get_load_snapshot()["safe_admit_capacity"] == 0

    def test_preempted_resident_request_keeps_capacity_until_finished(self) -> None:
        scheduler = self._make_scheduler(max_num_seqs=2)

        running = scheduler.add_request(_make_slo_step_request("running", reference_cost_ms=100, slo_ms=10000))
        first = scheduler.schedule()
        assert _new_ids(first) == [running]
        assert scheduler.update_from_output(first, _make_step_output(running, step_index=1)) == set()

        urgent = scheduler.add_request(_make_slo_step_request("urgent", reference_cost_ms=100, slo_ms=200, height=768))
        second = scheduler.schedule()
        assert _new_ids(second) == [urgent]
        assert scheduler.get_request_state(running).status == DiffusionRequestStatus.PREEMPTED
        assert scheduler.get_load_snapshot()["safe_admit_capacity"] == 0

        scheduler.update_from_output(second, _make_step_output(urgent, step_index=1))
        third = scheduler.add_request(
            _make_slo_step_request("third", reference_cost_ms=100, slo_ms=50, height=1024)
        )
        third_output = scheduler.schedule()

        assert _new_ids(third_output) == []
        assert third not in _cached_ids(third_output)
        assert scheduler.get_request_state(third).status == DiffusionRequestStatus.WAITING

    def test_none_key_waiting_request_respects_resident_capacity(self) -> None:
        scheduler = self._make_scheduler(max_num_seqs=2)

        running = scheduler.add_request(_make_slo_step_request("running", reference_cost_ms=100, slo_ms=10000))
        first = scheduler.schedule()
        assert _new_ids(first) == [running]
        assert scheduler.update_from_output(first, _make_step_output(running, step_index=1)) == set()

        urgent = scheduler.add_request(_make_slo_step_request("urgent", reference_cost_ms=100, slo_ms=200, height=768))
        second = scheduler.schedule()
        assert _new_ids(second) == [urgent]
        assert scheduler.get_request_state(running).status == DiffusionRequestStatus.PREEMPTED

        scheduler.update_from_output(second, _make_step_output(urgent, step_index=1))
        multi_prompt = scheduler.add_request(
            OmniDiffusionRequest(
                prompts=["multi-0", "multi-1"],
                sampling_params=OmniDiffusionSamplingParams(
                    num_inference_steps=4,
                    reference_cost_ms=100,
                    slo_ms=50,
                ),
                request_ids=["multi-0", "multi-1"],
            )
        )
        third = scheduler.schedule()

        assert multi_prompt not in third.scheduled_req_ids
        assert scheduler.get_request_state(multi_prompt).status == DiffusionRequestStatus.WAITING
        assert scheduler.get_load_snapshot()["safe_admit_capacity"] == 0

    def test_none_sampling_key_requests_are_singleton_buckets(self) -> None:
        scheduler = self._make_scheduler(max_num_seqs=2)

        req_a = scheduler.add_request(
            OmniDiffusionRequest(
                prompts=["a0", "a1"],
                sampling_params=OmniDiffusionSamplingParams(
                    num_inference_steps=4,
                    reference_cost_ms=100,
                    slo_ms=1000,
                ),
                request_ids=["a0", "a1"],
            )
        )
        req_b = scheduler.add_request(
            OmniDiffusionRequest(
                prompts=["b0", "b1"],
                sampling_params=OmniDiffusionSamplingParams(
                    num_inference_steps=4,
                    reference_cost_ms=100,
                    slo_ms=1000,
                ),
                request_ids=["b0", "b1"],
            )
        )

        sched_output = scheduler.schedule()

        assert len(_new_ids(sched_output)) == 1
        assert _new_ids(sched_output)[0] in {req_a, req_b}
        assert sched_output.num_waiting_reqs == 1

    def test_same_bucket_admits_most_urgent_waiting_requests(self) -> None:
        scheduler = self._make_scheduler(max_num_seqs=2)

        loose_1 = scheduler.add_request(_make_slo_step_request("loose-1", reference_cost_ms=100, slo_ms=10000))
        urgent = scheduler.add_request(_make_slo_step_request("urgent", reference_cost_ms=100, slo_ms=300))
        loose_2 = scheduler.add_request(_make_slo_step_request("loose-2", reference_cost_ms=100, slo_ms=10000))

        sched_output = scheduler.schedule()

        assert urgent in _new_ids(sched_output)
        assert len(_new_ids(sched_output)) == 2
        assert {loose_1, loose_2}.intersection(_new_ids(sched_output))

    def test_deadline_guard_skips_extra_requests_when_batch_would_miss(self) -> None:
        scheduler = self._make_scheduler(
            max_num_seqs=3,
            slo_config={
                "batch_growth_alpha": 5.0,
                "min_laxity_guard_ms": 0.0,
            },
        )

        urgent = scheduler.add_request(_make_slo_step_request("urgent", reference_cost_ms=300, slo_ms=350))
        scheduler.add_request(_make_slo_step_request("loose-1", reference_cost_ms=300, slo_ms=10000))
        scheduler.add_request(_make_slo_step_request("loose-2", reference_cost_ms=300, slo_ms=10000))

        sched_output = scheduler.schedule()

        assert _new_ids(sched_output) == [urgent]
        assert sched_output.num_waiting_reqs == 2

    def test_no_preemption_admission_guard_protects_running_bucket(self) -> None:
        scheduler = self._make_scheduler(
            max_num_seqs=2,
            slo_config={
                "default_step_ms": 100.0,
                "batch_growth_alpha": 2.0,
                "enable_step_preemption": False,
                "min_laxity_guard_ms": 0.0,
            },
        )

        running = scheduler.add_request(
            _make_slo_step_request(
                "running",
                reference_cost_ms=1000,
                slo_ms=10000,
                num_inference_steps=10,
            )
        )
        first = scheduler.schedule()
        assert _new_ids(first) == [running]
        assert scheduler.update_from_output(first, _make_step_output(running, step_index=1)) == set()

        urgent = scheduler.add_request(
            _make_slo_step_request(
                "urgent",
                reference_cost_ms=1000,
                slo_ms=200,
                num_inference_steps=10,
            )
        )
        second = scheduler.schedule()

        assert _new_ids(second) == []
        assert _cached_ids(second) == [running]
        assert scheduler.get_request_state(urgent).status == DiffusionRequestStatus.WAITING

    def test_no_preemption_admission_max_batch_size_caps_waiting_pack(self) -> None:
        scheduler = self._make_scheduler(
            max_num_seqs=4,
            slo_config={
                "enable_step_preemption": False,
                "no_preemption_admission_guard": True,
                "no_preemption_admission_max_batch_size": 2,
            },
        )

        req_ids = [
            scheduler.add_request(_make_slo_step_request(f"req-{idx}", reference_cost_ms=100, slo_ms=10000))
            for idx in range(4)
        ]

        sched_output = scheduler.schedule()

        assert _new_ids(sched_output) == req_ids[:2]
        assert sched_output.num_running_reqs == 2
        assert sched_output.num_waiting_reqs == 2
        assert scheduler.get_load_snapshot()["no_preemption_admission_max_batch_size"] == 2

    def test_same_key_batch_formation_preserves_absolute_deadline_guard_priority(self) -> None:
        scheduler = self._make_scheduler(
            max_num_seqs=2,
            slo_config={
                "batch_growth_alpha": 1.0,
                "min_laxity_guard_ms": 100.0,
            },
        )

        scheduler.add_request(
            _make_slo_step_request(
                "long-but-guard-safe",
                reference_cost_ms=2000,
                slo_ms=2150,
                num_inference_steps=10,
            )
        )
        urgent = scheduler.add_request(
            _make_slo_step_request(
                "short-inside-guard",
                reference_cost_ms=100,
                slo_ms=150,
                num_inference_steps=10,
            )
        )

        sched_output = scheduler.schedule()

        assert _new_ids(sched_output) == [urgent]

    def test_same_key_batch_formation_protects_feasible_request_from_missed_bucket(self) -> None:
        scheduler = self._make_scheduler(
            max_num_seqs=2,
            slo_config={
                "batch_growth_alpha": 2.0,
                "min_laxity_guard_ms": 100.0,
            },
        )

        missed = scheduler.add_request(
            _make_slo_step_request(
                "already-missed",
                reference_cost_ms=1000,
                slo_ms=900,
                num_inference_steps=10,
            )
        )
        scheduler.add_request(
            _make_slo_step_request(
                "feasible-alone",
                reference_cost_ms=100,
                slo_ms=250,
                num_inference_steps=10,
            )
        )

        sched_output = scheduler.schedule()

        assert _new_ids(sched_output) == [missed]

    def test_no_deadline_requests_preserve_step_scheduler_fifo(self) -> None:
        scheduler = self._make_scheduler(max_num_seqs=3)

        req_a = scheduler.add_request(_make_step_request("a"))
        req_b = scheduler.add_request(
            _make_step_request(
                "b",
                sampling_params=OmniDiffusionSamplingParams(
                    height=768,
                    num_inference_steps=4,
                ),
            )
        )
        scheduler.add_request(_make_step_request("c"))

        first = scheduler.schedule()

        assert _new_ids(first) == [req_a]
        assert first.num_waiting_reqs == 2

        scheduler.update_from_output(first, _make_step_output(req_a, step_index=4, finished=True))
        second = scheduler.schedule()

        assert _new_ids(second) == [req_b]
        assert second.num_waiting_reqs == 1

    def test_load_snapshot_exposes_bucket_cost_and_capacity(self) -> None:
        scheduler = self._make_scheduler(max_num_seqs=2)
        scheduler.add_request(_make_slo_step_request("a", reference_cost_ms=100, slo_ms=1000))
        scheduler.add_request(_make_slo_step_request("b", reference_cost_ms=100, slo_ms=1000, height=768))

        snapshot = scheduler.get_load_snapshot()

        assert snapshot["policy"] == "SloStepScheduler"
        assert snapshot["safe_admit_capacity"] == 2
        assert len(snapshot["buckets"]) == 2
        assert all(bucket["estimated_step_ms"] > 0 for bucket in snapshot["buckets"])

    def test_cost_model_table_replaces_reference_cost_estimate(self, tmp_path) -> None:
        model_path = tmp_path / "step_cost_model.json"
        model_path.write_text(
            json.dumps(
                {
                    "table_lookup": {
                        "Qwen/Qwen-Image": {
                            "1024x1024x1": {
                                "1": {
                                    "1.0": {
                                        "effective_batch_size": 1.0,
                                        "denoise_step_ms": 100.0,
                                        "p90_ms": 123.0,
                                    }
                                },
                                "2": {
                                    "2.0": {
                                        "effective_batch_size": 2.0,
                                        "denoise_step_ms": 180.0,
                                        "p90_ms": 210.0,
                                    }
                                },
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        scheduler = self._make_scheduler(
            max_num_seqs=2,
            slo_config={
                "step_cost_model_path": str(model_path),
                "step_cost_metric": "p90_ms",
            },
        )
        scheduler.add_request(
            _make_slo_step_request("a", reference_cost_ms=10, slo_ms=100000, num_inference_steps=50, height=1024)
        )
        scheduler.add_request(
            _make_slo_step_request("b", reference_cost_ms=10, slo_ms=100000, num_inference_steps=50, height=1024)
        )

        snapshot = scheduler.get_load_snapshot()

        bucket = snapshot["buckets"][0]
        assert bucket["estimated_step_ms"] == pytest.approx(210.0)
        assert bucket["step_cost_source"] == "exact_table"
        assert bucket["latent_tokens"] == 4096
        assert bucket["incremental_step_ms_if_add_one"] >= 0.0

    def test_cost_model_named_formula_is_used_when_table_misses(self) -> None:
        scheduler = self._make_scheduler(
            max_num_seqs=1,
            slo_config={
                "step_cost_formula": "qwen_image_910b_tp2_v1",
                "step_cost_safety_factor": 1.1,
            },
        )
        scheduler.add_request(
            _make_slo_step_request("a", reference_cost_ms=10, slo_ms=1000, num_inference_steps=50, height=1024)
        )

        snapshot = scheduler.get_load_snapshot()

        bucket = snapshot["buckets"][0]
        assert bucket["step_cost_source"] == "latent_formula"
        assert bucket["estimated_step_ms"] == pytest.approx(
            1.1 * (288.68 - 230.01 - 38.12 + 335.52 + 82.63),
        )

    def test_cost_model_uses_actual_batch_size_not_effective_batch_size(self, tmp_path) -> None:
        model_path = tmp_path / "step_cost_model.json"
        model_path.write_text(
            json.dumps(
                {
                    "table_lookup": {
                        "Qwen/Qwen-Image": {
                            "1024x1024x1": {
                                "1": {
                                    "2.0": {
                                        "effective_batch_size": 2.0,
                                        "denoise_step_ms": 111.0,
                                        "p90_ms": 111.0,
                                    }
                                },
                                "2": {
                                    "2.0": {
                                        "effective_batch_size": 2.0,
                                        "denoise_step_ms": 222.0,
                                        "p90_ms": 222.0,
                                    }
                                },
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        scheduler = self._make_scheduler(
            max_num_seqs=1,
            slo_config={
                "step_cost_model_path": str(model_path),
                "step_cost_metric": "p90_ms",
            },
        )
        scheduler.add_request(
            OmniDiffusionRequest(
                prompts=[{"prompt": "cfg", "negative_prompt": "avoid blur"}],
                sampling_params=OmniDiffusionSamplingParams(
                    height=1024,
                    width=1024,
                    num_inference_steps=50,
                    reference_cost_ms=10,
                    slo_ms=100000,
                    true_cfg_scale=4.0,
                ),
                request_ids=["cfg"],
            )
        )

        snapshot = scheduler.get_load_snapshot()

        assert snapshot["buckets"][0]["effective_batch_size"] == 2.0
        assert snapshot["buckets"][0]["estimated_step_ms"] == pytest.approx(111.0)

    def test_true_cfg_effective_batch_requires_negative_prompt(self, tmp_path) -> None:
        model_path = tmp_path / "step_cost_model.json"
        model_path.write_text(
            json.dumps(
                {
                    "table_lookup": {
                        "Qwen/Qwen-Image": {
                            "1024x1024x1": {
                                "1": {
                                    "1.0": {
                                        "effective_batch_size": 1.0,
                                        "denoise_step_ms": 111.0,
                                        "p90_ms": 111.0,
                                    },
                                    "2.0": {
                                        "effective_batch_size": 2.0,
                                        "denoise_step_ms": 222.0,
                                        "p90_ms": 222.0,
                                    },
                                }
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        scheduler = self._make_scheduler(
            max_num_seqs=1,
            slo_config={
                "step_cost_model_path": str(model_path),
                "step_cost_metric": "p90_ms",
            },
        )
        scheduler.add_request(
            OmniDiffusionRequest(
                prompts=[{"prompt": "plain"}],
                sampling_params=OmniDiffusionSamplingParams(
                    height=1024,
                    width=1024,
                    num_inference_steps=50,
                    reference_cost_ms=10,
                    slo_ms=100000,
                    true_cfg_scale=4.0,
                ),
                request_ids=["plain"],
            )
        )

        snapshot = scheduler.get_load_snapshot()

        assert snapshot["buckets"][0]["effective_batch_size"] == 1.0
        assert snapshot["buckets"][0]["estimated_step_ms"] == pytest.approx(111.0)

    def test_guidance_scale_flag_does_not_double_qwen_effective_batch_without_true_cfg(self, tmp_path) -> None:
        model_path = tmp_path / "step_cost_model.json"
        model_path.write_text(
            json.dumps(
                {
                    "table_lookup": {
                        "Qwen/Qwen-Image": {
                            "1024x1024x1": {
                                "1": {
                                    "1.0": {
                                        "effective_batch_size": 1.0,
                                        "denoise_step_ms": 111.0,
                                        "p90_ms": 111.0,
                                    },
                                    "2.0": {
                                        "effective_batch_size": 2.0,
                                        "denoise_step_ms": 222.0,
                                        "p90_ms": 222.0,
                                    },
                                }
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        scheduler = self._make_scheduler(
            max_num_seqs=1,
            slo_config={
                "step_cost_model_path": str(model_path),
                "step_cost_metric": "p90_ms",
            },
        )
        scheduler.add_request(
            OmniDiffusionRequest(
                prompts=[{"prompt": "plain", "negative_prompt": "avoid blur"}],
                sampling_params=OmniDiffusionSamplingParams(
                    height=1024,
                    width=1024,
                    num_inference_steps=50,
                    reference_cost_ms=10,
                    slo_ms=100000,
                    guidance_scale=7.5,
                    true_cfg_scale=1.0,
                ),
                request_ids=["plain"],
            )
        )

        snapshot = scheduler.get_load_snapshot()

        assert scheduler.get_request_state("plain").req.sampling_params.do_classifier_free_guidance
        assert snapshot["buckets"][0]["effective_batch_size"] == 1.0
        assert snapshot["buckets"][0]["estimated_step_ms"] == pytest.approx(111.0)

    def test_cost_model_effective_batch_counts_prompts_in_single_request(self, tmp_path) -> None:
        model_path = tmp_path / "step_cost_model.json"
        model_path.write_text(
            json.dumps(
                {
                    "table_lookup": {
                        "Qwen/Qwen-Image": {
                            "1024x1024x1": {
                                "1": {
                                    "1.0": {
                                        "effective_batch_size": 1.0,
                                        "denoise_step_ms": 111.0,
                                        "p90_ms": 111.0,
                                    },
                                    "2.0": {
                                        "effective_batch_size": 2.0,
                                        "denoise_step_ms": 222.0,
                                        "p90_ms": 222.0,
                                    },
                                }
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        scheduler = self._make_scheduler(
            max_num_seqs=1,
            slo_config={
                "step_cost_model_path": str(model_path),
                "step_cost_metric": "p90_ms",
            },
        )
        scheduler.add_request(
            OmniDiffusionRequest(
                prompts=["p0", "p1"],
                sampling_params=OmniDiffusionSamplingParams(
                    height=1024,
                    width=1024,
                    num_inference_steps=50,
                    reference_cost_ms=10,
                    slo_ms=100000,
                ),
                request_ids=["p0", "p1"],
            )
        )

        snapshot = scheduler.get_load_snapshot()

        assert snapshot["buckets"][0]["effective_batch_size"] == 2.0
        assert snapshot["buckets"][0]["estimated_step_ms"] == pytest.approx(222.0)

    def test_json_fallback_formula_is_model_scoped(self, tmp_path) -> None:
        model_path = tmp_path / "step_cost_model.json"
        model_path.write_text(
            json.dumps(
                {
                    "table_lookup": {
                        "Qwen/Qwen-Image": {
                            "1024x1024x1": {
                                "1": {
                                    "1.0": {
                                        "effective_batch_size": 1.0,
                                        "denoise_step_ms": 100.0,
                                        "p90_ms": 100.0,
                                    }
                                }
                            }
                        }
                    },
                    "fallback": {
                        "coefficients": {
                            "c0": 9999.0,
                            "c1": 0.0,
                            "c2": 0.0,
                        },
                        "gamma": 1.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        scheduler = self._make_scheduler(
            max_num_seqs=1,
            model="Other/Model",
            slo_config={
                "step_cost_model_path": str(model_path),
                "step_cost_metric": "p90_ms",
            },
        )
        scheduler.add_request(
            _make_slo_step_request("a", reference_cost_ms=5000, slo_ms=100000, num_inference_steps=50, height=1024)
        )

        snapshot = scheduler.get_load_snapshot()

        assert snapshot["buckets"][0]["step_cost_source"] == "legacy"
        assert snapshot["buckets"][0]["estimated_step_ms"] == pytest.approx(100.0)

    def test_named_formula_is_model_scoped(self) -> None:
        scheduler = self._make_scheduler(
            max_num_seqs=1,
            model="Other/Model",
            slo_config={
                "step_cost_formula": "qwen_image_910b_tp2_v1",
            },
        )
        scheduler.add_request(
            _make_slo_step_request("a", reference_cost_ms=5000, slo_ms=100000, num_inference_steps=50, height=1024)
        )

        snapshot = scheduler.get_load_snapshot()

        assert snapshot["buckets"][0]["step_cost_source"] == "legacy"
        assert snapshot["buckets"][0]["estimated_step_ms"] == pytest.approx(100.0)

    def test_cost_model_does_not_use_default_resolution_as_shape(self, tmp_path) -> None:
        model_path = tmp_path / "step_cost_model.json"
        model_path.write_text(
            json.dumps(
                {
                    "table_lookup": {
                        "Qwen/Qwen-Image": {
                            "640x640x1": {
                                "1": {
                                    "1.0": {
                                        "effective_batch_size": 1.0,
                                        "denoise_step_ms": 999.0,
                                        "p90_ms": 999.0,
                                    }
                                }
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        scheduler = self._make_scheduler(
            max_num_seqs=1,
            slo_config={
                "step_cost_model_path": str(model_path),
                "step_cost_metric": "p90_ms",
            },
        )
        scheduler.add_request(
            _make_slo_step_request("a", reference_cost_ms=5000, slo_ms=100000, num_inference_steps=50)
        )

        snapshot = scheduler.get_load_snapshot()

        assert snapshot["buckets"][0]["step_cost_source"] == "legacy"
        assert snapshot["buckets"][0]["estimated_step_ms"] == pytest.approx(100.0)


class TestTokenSloStepScheduler:
    @staticmethod
    def _qwen_dynamic_config(slo_config: dict | None = None, max_num_seqs: int = 4) -> SimpleNamespace:
        base_slo_config = {
            "default_step_ms": 100.0,
            "batch_growth_alpha": 1.0,
            "ignore_request_reference_cost": True,
            "use_shape_fallback_cost": True,
        }
        if slo_config:
            base_slo_config.update(slo_config)
        return SimpleNamespace(
            model="Qwen/Qwen-Image",
            model_class_name="QwenImagePipeline",
            step_execution=True,
            max_num_seqs=max_num_seqs,
            enforce_eager=True,
            cache_backend="none",
            diffusion_kv_cache_dtype=None,
            parallel_config=SimpleNamespace(
                sequence_parallel_size=1,
                ring_degree=1,
                cfg_parallel_size=1,
                use_hsdp=False,
            ),
            additional_config={
                "diffusion_dynamic_step_batching_enabled": True,
                "diffusion_slo_scheduler": base_slo_config,
            },
        )

    def _make_scheduler(
        self,
        max_num_seqs: int = 4,
        slo_config: dict | None = None,
        scheduler_cls=TokenSloStepScheduler,
    ) -> TokenSloStepScheduler:
        scheduler = scheduler_cls()
        scheduler.initialize(self._qwen_dynamic_config(slo_config=slo_config, max_num_seqs=max_num_seqs))
        return scheduler

    @staticmethod
    def _make_token_request(
        req_id: str,
        *,
        height: int,
        slo_ms: float = 10000.0,
        num_inference_steps: int = 10,
        deadline_time_s: float | None = None,
        arrival_time_s: float | None = None,
    ) -> OmniDiffusionRequest:
        kwargs = {
            "height": height,
            "width": height,
            "num_inference_steps": num_inference_steps,
            "reference_cost_ms": 1000.0,
            "slo_ms": slo_ms,
            "true_cfg_scale": 1.0,
        }
        if arrival_time_s is not None:
            kwargs["arrival_time_s"] = arrival_time_s
        if deadline_time_s is not None:
            kwargs["deadline_time_s"] = deadline_time_s
        sampling_params = OmniDiffusionSamplingParams(**kwargs)
        return OmniDiffusionRequest(
            prompts=[{"prompt": f"prompt-{req_id}"}],
            sampling_params=sampling_params,
            request_id=req_id,
        )

    def test_mixed_dynamic_batch_uses_token_weighted_effective_size(self) -> None:
        scheduler = self._make_scheduler(max_num_seqs=4)
        scheduler.add_request(self._make_token_request("small", height=512))
        scheduler.add_request(self._make_token_request("large", height=1024))

        snapshot = scheduler.get_load_snapshot()

        assert snapshot["policy"] == "TokenSloStepScheduler"
        assert len(snapshot["buckets"]) == 1
        bucket = snapshot["buckets"][0]
        assert bucket["candidate_batch_size"] == 2
        assert bucket["max_latent_tokens"] == 4096
        assert bucket["total_latent_tokens"] == 5120
        assert bucket["token_effective_batch_size"] == pytest.approx(1.25)
        assert bucket["token_batch_utilization"] == pytest.approx(0.625)
        assert bucket["estimated_step_ms"] == pytest.approx(125.0)
        assert snapshot["token_pressure"] > 0

    def test_incremental_step_cost_uses_largest_token_shape_with_cost_model(self) -> None:
        class RecordingCostModel:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def latent_tokens(self, width, height, num_frames=1):
                return int((float(width) / 16.0) * (float(height) / 16.0) * float(num_frames or 1))

            def estimate(self, **kwargs):
                self.calls.append(dict(kwargs))
                return SimpleNamespace(
                    step_ms=float(kwargs["effective_batch_size"]) * 100.0,
                    source="recording",
                    latent_tokens=self.latent_tokens(
                        kwargs["width"],
                        kwargs["height"],
                        kwargs.get("num_frames", 1),
                    ),
                )

        scheduler = self._make_scheduler(max_num_seqs=4)
        recorder = RecordingCostModel()
        scheduler.cost_model = recorder
        scheduler.add_request(self._make_token_request("small", height=512))
        scheduler.add_request(self._make_token_request("large", height=1024))

        bucket = scheduler.get_load_snapshot()["buckets"][0]

        assert bucket["estimated_step_ms"] == pytest.approx(125.0)
        assert bucket["estimated_step_ms_if_add_one"] == pytest.approx(225.0)
        assert bucket["incremental_step_ms_if_add_one"] == pytest.approx(100.0)
        assert recorder.calls[-1]["width"] == 1024
        assert recorder.calls[-1]["height"] == 1024
        assert recorder.calls[-1]["batch_size"] == 3
        assert recorder.calls[-1]["effective_batch_size"] == pytest.approx(2.25)

    def test_snapshot_token_fields_follow_actual_candidate_batch(self) -> None:
        scheduler = self._make_scheduler(max_num_seqs=1)
        large_loose = scheduler.add_request(self._make_token_request("large-loose", height=1024, slo_ms=10000.0))
        small_urgent = scheduler.add_request(self._make_token_request("small-urgent", height=512, slo_ms=300.0))

        snapshot = scheduler.get_load_snapshot()

        assert len(snapshot["buckets"]) == 1
        bucket = snapshot["buckets"][0]
        assert bucket["sched_req_ids"] == [small_urgent]
        assert large_loose not in bucket["sched_req_ids"]
        assert bucket["candidate_batch_size"] == 1
        assert bucket["max_latent_tokens"] == 1024
        assert bucket["total_latent_tokens"] == 1024

    def test_no_preemption_token_guard_blocks_large_admission_that_hurts_resident(self) -> None:
        scheduler = self._make_scheduler(max_num_seqs=2, slo_config={"min_laxity_guard_ms": 0.0})
        running = scheduler.add_request(self._make_token_request("running", height=512, slo_ms=300.0))
        first = scheduler.schedule()
        assert _new_ids(first) == [running]
        assert scheduler.update_from_output(first, _make_step_output(running, step_index=1)) == set()

        large = scheduler.add_request(self._make_token_request("large", height=1024, slo_ms=10000.0))
        second = scheduler.schedule()

        assert _new_ids(second) == []
        assert _cached_ids(second) == [running]
        assert scheduler.get_request_state(large).status == DiffusionRequestStatus.WAITING

    def test_token_policy_forces_no_preemption_even_if_config_enables_it(self) -> None:
        scheduler = self._make_scheduler(
            max_num_seqs=1,
            slo_config={"enable_step_preemption": True},
        )
        assert scheduler.enable_step_preemption is False

        running = scheduler.add_request(self._make_token_request("running", height=512, slo_ms=10000.0))
        first = scheduler.schedule()
        assert _new_ids(first) == [running]
        assert scheduler.update_from_output(first, _make_step_output(running, step_index=1)) == set()

        urgent = scheduler.add_request(
            OmniDiffusionRequest(
                prompts=[{"prompt": "urgent", "negative_prompt": "bad"}],
                sampling_params=OmniDiffusionSamplingParams(
                    height=1024,
                    width=1024,
                    num_inference_steps=10,
                    reference_cost_ms=1000.0,
                    slo_ms=100.0,
                    true_cfg_scale=4.0,
                ),
                request_id="urgent",
            )
        )

        second = scheduler.schedule()

        assert _new_ids(second) == []
        assert _cached_ids(second) == [running]
        assert scheduler.get_request_state(urgent).status == DiffusionRequestStatus.WAITING


class TestAdaptiveTokenSloStepScheduler:
    def _make_scheduler(
        self,
        max_num_seqs: int = 4,
        slo_config: dict | None = None,
    ) -> AdaptiveTokenSloStepScheduler:
        scheduler = AdaptiveTokenSloStepScheduler()
        scheduler.initialize(
            TestTokenSloStepScheduler._qwen_dynamic_config(
                slo_config=slo_config,
                max_num_seqs=max_num_seqs,
            )
        )
        return scheduler

    _make_token_request = staticmethod(TestTokenSloStepScheduler._make_token_request)

    def test_adaptive_guard_blocks_low_ratio_admission_for_resident(self) -> None:
        scheduler = self._make_scheduler(
            max_num_seqs=2,
            slo_config={
                "adaptive_min_laxity_ratio": 0.20,
                "min_laxity_guard_ms": 0.0,
            },
        )
        running = scheduler.add_request(self._make_token_request("large-running", height=1024, slo_ms=1200.0))
        first = scheduler.schedule()
        assert _new_ids(first) == [running]
        assert scheduler.update_from_output(first, _make_step_output(running, step_index=1)) == set()

        small = scheduler.add_request(self._make_token_request("small-waiting", height=512, slo_ms=10000.0))
        second = scheduler.schedule()

        assert _new_ids(second) == []
        assert _cached_ids(second) == [running]
        assert scheduler.get_request_state(small).status == DiffusionRequestStatus.WAITING

    def test_adaptive_priority_prefers_large_token_work_inside_laxity_window(self) -> None:
        now_s = time.time()
        scheduler = self._make_scheduler(
            max_num_seqs=1,
            slo_config={
                "adaptive_priority_laxity_window_ms": 500.0,
                "adaptive_priority_ratio_window": 0.25,
            },
        )
        small = scheduler.add_request(
            self._make_token_request(
                "small",
                height=512,
                arrival_time_s=now_s,
                deadline_time_s=now_s + 0.95,
            )
        )
        large = scheduler.add_request(
            self._make_token_request(
                "large",
                height=1024,
                arrival_time_s=now_s,
                deadline_time_s=now_s + 1.85,
            )
        )

        scheduled = scheduler.schedule()

        assert _new_ids(scheduled) == [large]
        assert scheduler.get_request_state(small).status == DiffusionRequestStatus.WAITING

    def test_adaptive_snapshot_includes_laxity_ratio(self) -> None:
        scheduler = self._make_scheduler(max_num_seqs=2)
        scheduler.add_request(self._make_token_request("small", height=512))
        scheduler.add_request(self._make_token_request("large", height=1024))

        snapshot = scheduler.get_load_snapshot()

        assert snapshot["policy"] == "AdaptiveTokenSloStepScheduler"
        assert "min_laxity_ratio" in snapshot["buckets"][0]

    def test_adaptive_does_not_preempt_running_bucket_for_urgent_different_key(self) -> None:
        scheduler = self._make_scheduler(max_num_seqs=2)
        running = scheduler.add_request(self._make_token_request("running", height=512, slo_ms=10000.0))
        first = scheduler.schedule()
        assert _new_ids(first) == [running]
        assert scheduler.update_from_output(first, _make_step_output(running, step_index=1)) == set()

        urgent = scheduler.add_request(
            OmniDiffusionRequest(
                prompts=[{"prompt": "urgent", "negative_prompt": "bad"}],
                sampling_params=OmniDiffusionSamplingParams(
                    height=1024,
                    width=1024,
                    num_inference_steps=10,
                    reference_cost_ms=1000.0,
                    slo_ms=100.0,
                    true_cfg_scale=4.0,
                ),
                request_id="urgent",
            )
        )
        assert (
            scheduler.get_request_state(running).sampling_params_key
            != scheduler.get_request_state(urgent).sampling_params_key
        )
        second = scheduler.schedule()

        assert _new_ids(second) == []
        assert _cached_ids(second) == [running]
        assert scheduler.get_request_state(urgent).status == DiffusionRequestStatus.WAITING


class TestTokenStepPreemptiveSloStepScheduler:
    def _make_scheduler(
        self,
        max_num_seqs: int = 1,
        slo_config: dict | None = None,
    ) -> TokenStepPreemptiveSloStepScheduler:
        scheduler = TokenStepPreemptiveSloStepScheduler()
        scheduler.initialize(
            TestTokenSloStepScheduler._qwen_dynamic_config(
                slo_config=slo_config,
                max_num_seqs=max_num_seqs,
            )
        )
        return scheduler

    _make_token_request = staticmethod(TestTokenSloStepScheduler._make_token_request)

    def test_step_boundary_preempts_large_for_urgent_small_and_resumes_large(self) -> None:
        now_s = time.time()
        scheduler = self._make_scheduler(max_num_seqs=1)
        large = scheduler.add_request(
            self._make_token_request(
                "large",
                height=1024,
                arrival_time_s=now_s,
                deadline_time_s=now_s + 10.0,
            )
        )
        first = scheduler.schedule()
        assert _new_ids(first) == [large]
        assert scheduler.update_from_output(first, _make_step_output(large, step_index=1)) == set()

        small = scheduler.add_request(
            self._make_token_request(
                "small",
                height=512,
                arrival_time_s=now_s + 0.01,
                deadline_time_s=now_s + 0.30,
            )
        )
        assert (
            scheduler.get_request_state(large).sampling_params_key
            == scheduler.get_request_state(small).sampling_params_key
        )

        second = scheduler.schedule()

        assert _new_ids(second) == [small]
        assert _cached_ids(second) == []
        assert scheduler.get_request_state(large).status == DiffusionRequestStatus.PREEMPTED
        assert second.debug_info["policy"] == "token_step_preemptive_slo"
        assert second.debug_info["step_preemption_count"] == 1
        assert second.debug_info["preempted_req_ids"] == [large]
        assert scheduler.get_load_snapshot()["preempted_waiting_req_ids"] == [large]

        assert scheduler.update_from_output(second, _make_step_output(small, step_index=1, finished=True)) == {small}
        third = scheduler.schedule()

        assert _new_ids(third) == []
        assert _cached_ids(third) == [large]
        assert scheduler.get_request_state(large).status == DiffusionRequestStatus.RUNNING
        assert third.debug_info["resumed_preempted_req_ids"] == [large]
        assert third.debug_info["resumed_preempted_extra_wait_ms"][large] >= 0.0

    def test_step_preemptive_snapshot_reports_policy_knobs(self) -> None:
        scheduler = self._make_scheduler(
            max_num_seqs=1,
            slo_config={"allow_preemptive_admission_swap": False},
        )
        snapshot = scheduler.get_load_snapshot()

        assert snapshot["policy"] == "TokenStepPreemptiveSloStepScheduler"
        assert snapshot["enable_step_preemption"] is True
        assert snapshot["allow_preemptive_admission_swap"] is False

    def test_step_preemptive_can_preserve_resident_capacity_guard(self) -> None:
        now_s = time.time()
        scheduler = self._make_scheduler(
            max_num_seqs=1,
            slo_config={"allow_preemptive_admission_swap": False},
        )
        large = scheduler.add_request(
            self._make_token_request(
                "large",
                height=1024,
                arrival_time_s=now_s,
                deadline_time_s=now_s + 10.0,
            )
        )
        first = scheduler.schedule()
        assert _new_ids(first) == [large]
        assert scheduler.update_from_output(first, _make_step_output(large, step_index=1)) == set()

        small = scheduler.add_request(
            self._make_token_request(
                "small",
                height=512,
                arrival_time_s=now_s + 0.01,
                deadline_time_s=now_s + 0.30,
            )
        )
        second = scheduler.schedule()

        assert _new_ids(second) == []
        assert _cached_ids(second) == [large]
        assert scheduler.get_request_state(small).status == DiffusionRequestStatus.WAITING
        assert second.debug_info["step_preemption_count"] == 0
