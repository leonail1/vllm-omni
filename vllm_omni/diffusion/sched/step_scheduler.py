# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import _BaseScheduler
from vllm_omni.diffusion.sched.step_cost_model import DiffusionStepCostModel, estimate_request_effective_size
from vllm_omni.diffusion.sched.interface import (
    DiffusionRequestState,
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    SamplingParamsKey,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import RunnerOutput

logger = init_logger(__name__)


@dataclass
class _StepProgress:
    current_step: int
    total_steps: int


class StepScheduler(_BaseScheduler):
    """Placeholder scheduler that advances a request one denoise step per update."""

    def __init__(self) -> None:
        super().__init__()
        self._request_progress: dict[str, _StepProgress] = {}
        self.default_step_ms = 1.0
        self.batch_growth_alpha = 0.60
        self.decode_ms = 0.0
        self.cost_model: DiffusionStepCostModel | None = None
        self.use_shape_fallback_cost = True

    def initialize(self, od_config) -> None:
        super().initialize(od_config)
        slo_config = _get_slo_scheduler_config(od_config)
        self.default_step_ms = _coerce_positive_float(slo_config.get("default_step_ms"), 1.0)
        self.batch_growth_alpha = _coerce_nonnegative_float(slo_config.get("batch_growth_alpha"), 0.60)
        self.decode_ms = _coerce_nonnegative_float(slo_config.get("decode_ms"), 0.0)
        self.use_shape_fallback_cost = _coerce_bool(slo_config.get("use_shape_fallback_cost"), True)
        self.cost_model = DiffusionStepCostModel.from_config(
            slo_config,
            default_step_ms=self.default_step_ms,
        )

    def _reset_scheduler_state(self) -> None:
        self._request_progress.clear()

    def add_request(self, request: OmniDiffusionRequest) -> str:
        sched_req_id = self._make_sched_req_id(request)
        total_steps = self._get_total_steps(request)
        if total_steps <= 0:
            raise ValueError(f"Diffusion request {sched_req_id} must have positive total_steps, got {total_steps}")

        current_step = request.sampling_params.step_index or 0
        if current_step < 0 or current_step >= total_steps:
            raise ValueError(
                f"Diffusion request {sched_req_id} has invalid initial step_index {current_step} "
                f"for total_steps={total_steps}"
            )

        request.sampling_params.step_index = current_step
        sched_req_id = self._add_request_with_sched_req_id(sched_req_id, request)
        self._request_progress[sched_req_id] = _StepProgress(current_step=current_step, total_steps=total_steps)
        logger.debug(
            "StepScheduler add_request: %s (step=%d/%d, waiting=%d)",
            sched_req_id,
            current_step,
            total_steps,
            len(self._waiting),
        )
        return sched_req_id

    def schedule(self) -> DiffusionSchedulerOutput:
        return super().schedule()

    def get_load_snapshot(self) -> dict[str, Any]:
        now_s = time.time()
        buckets = []
        for key, states in self._snapshot_candidate_batches():
            step_estimate = self._estimate_step_cost(states)
            representative = states[0]
            sampling = representative.req.sampling_params
            width, height = _sampling_dimensions(sampling)
            frames = _coerce_float(getattr(sampling, "num_frames", None)) or 1.0
            buckets.append(
                {
                    "key": None if key is None else asdict(key),
                    "num_running": sum(1 for state in states if state.sched_req_id in self._running),
                    "num_waiting": sum(1 for state in states if state.sched_req_id in self._waiting),
                    "candidate_batch_size": len(states),
                    "effective_batch_size": self._effective_batch_size(states),
                    "estimated_step_ms": step_estimate.step_ms,
                    "estimated_step_ms_if_add_one": step_estimate.step_ms
                    + self._estimate_incremental_step_ms(states, 1.0),
                    "incremental_step_ms_if_add_one": self._estimate_incremental_step_ms(states, 1.0),
                    "step_cost_source": step_estimate.source,
                    "latent_tokens": step_estimate.latent_tokens,
                    "shape": {
                        "width": width,
                        "height": height,
                        "num_frames": frames,
                    },
                    "min_laxity_ms": self._bucket_min_laxity_ms(states, step_estimate.step_ms, now_s),
                    "min_remaining_steps": min(self._remaining_steps(state) for state in states),
                    "max_remaining_steps": max(self._remaining_steps(state) for state in states),
                    "oldest_arrival_time_s": min(state.arrival_time_s for state in states),
                }
            )
        return {
            "policy": self.__class__.__name__,
            "timestamp_s": now_s,
            "num_waiting": len(self._waiting),
            "num_running": len(self._running),
            "max_num_running": self.max_num_running_reqs,
            "safe_admit_capacity": max(0, self.max_num_running_reqs - len(self._running)),
            "buckets": buckets,
        }

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: RunnerOutput) -> set[str]:
        scheduled_req_ids = sched_output.scheduled_req_ids
        if not scheduled_req_ids:
            return set()

        terminal_statuses: dict[str, DiffusionRequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}
        for sched_req_id in scheduled_req_ids:
            state = self._request_states.get(sched_req_id)
            progress = self._request_progress.get(sched_req_id)
            if state is None or progress is None or state.is_finished():
                continue
            req_output = output.get_req_output(sched_req_id)
            if req_output is None:
                logger.warning(
                    "No RunnerOutput for request %s, treating as error",
                    sched_req_id,
                )
                terminal_statuses[sched_req_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[sched_req_id] = "No output for request"
                continue

            req_result = req_output.result
            output_error = req_result.error if req_result is not None else None
            if output_error is not None:
                terminal_statuses[sched_req_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[sched_req_id] = output_error
                continue

            if req_output.step_index is None:
                logger.warning(
                    "Received RunnerOutput with no step_index for request %s, treating as error",
                    sched_req_id,
                )
                terminal_statuses[sched_req_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[sched_req_id] = "Missing step_index in RunnerOutput"
                continue

            # We assume that the decoding stage is executed immediately after the denoising stage completes.
            progress.current_step = req_output.step_index
            state.req.sampling_params.step_index = req_output.step_index
            if req_output.finished:
                terminal_statuses[sched_req_id] = DiffusionRequestStatus.FINISHED_COMPLETED
                terminal_errors[sched_req_id] = None
            else:
                state.error = None

        return self._finalize_update_from_output(sched_output, terminal_statuses, terminal_errors)

    def _pop_extra_request_state(self, sched_req_id: str) -> None:
        self._request_progress.pop(sched_req_id, None)

    def _get_total_steps(self, request: OmniDiffusionRequest) -> int:
        sampling = request.sampling_params

        if sampling.timesteps is not None:
            return self._sequence_length(sampling.timesteps)
        if sampling.sigmas is not None:
            return len(sampling.sigmas)
        return int(sampling.num_inference_steps)

    @staticmethod
    def _sequence_length(values: Any) -> int:
        ndim = getattr(values, "ndim", None)
        if ndim == 0:
            return 1

        shape = getattr(values, "shape", None)
        if shape is not None:
            return int(shape[0])

        return len(values)

    def _snapshot_candidate_batches(self) -> list[tuple[SamplingParamsKey | None, list[DiffusionRequestState]]]:
        grouped: dict[SamplingParamsKey, list[DiffusionRequestState]] = defaultdict(list)
        singleton_buckets: list[tuple[SamplingParamsKey | None, list[DiffusionRequestState]]] = []
        seen_req_ids: set[str] = set()

        for sched_req_id in [*self._running, *self._waiting]:
            if sched_req_id in seen_req_ids:
                continue
            seen_req_ids.add(sched_req_id)
            state = self._request_states.get(sched_req_id)
            if state is None or state.is_finished():
                continue
            key = state.sampling_params_key
            if key is None:
                singleton_buckets.append((None, [state]))
            else:
                grouped[key].append(state)

        buckets: list[tuple[SamplingParamsKey | None, list[DiffusionRequestState]]] = []
        for key, states in grouped.items():
            selected = states[: self.max_num_running_reqs]
            if selected:
                buckets.append((key, selected))
        buckets.extend(singleton_buckets)
        return buckets

    def _bucket_min_laxity_ms(self, states: list[DiffusionRequestState], step_ms: float, now_s: float) -> float:
        laxities = []
        for state in states:
            if state.deadline_time_s is None:
                continue
            remaining_work_ms = self._remaining_steps(state) * step_ms + self.decode_ms
            laxities.append((state.deadline_time_s - now_s) * 1000.0 - remaining_work_ms)
        return min(laxities) if laxities else math.inf

    def _remaining_steps(self, state: DiffusionRequestState) -> int:
        progress = self._request_progress.get(state.sched_req_id)
        if progress is None:
            return 1
        return max(progress.total_steps - progress.current_step, 1)

    def _estimate_step_cost(self, states: list[DiffusionRequestState]):
        if states and self.cost_model is not None:
            sampling = states[0].req.sampling_params
            width, height = _sampling_dimensions(sampling)
            estimate = self.cost_model.estimate(
                model=getattr(self.od_config, "model", None),
                width=width,
                height=height,
                num_frames=getattr(sampling, "num_frames", 1),
                batch_size=len(states),
                effective_batch_size=self._effective_batch_size(states),
            )
            if estimate.step_ms > 0 and estimate.source != "default":
                return estimate

        single_step_candidates: list[float] = []
        for state in states:
            progress = self._request_progress.get(state.sched_req_id)
            if state.reference_cost_ms is not None and state.reference_cost_ms > 0 and progress is not None:
                single_step_candidates.append(max(state.reference_cost_ms / max(progress.total_steps, 1), 0.001))
                continue
            single_step_candidates.append(self._shape_fallback_step_ms(state))

        single_step_ms = max(single_step_candidates) if single_step_candidates else self.default_step_ms
        b_eff = self._effective_batch_size(states)
        step_ms = max(single_step_ms * (1.0 + self.batch_growth_alpha * max(b_eff - 1.0, 0.0)), 0.001)
        return _StepSnapshotCost(step_ms, "legacy", None)

    def _estimate_incremental_step_ms(self, states: list[DiffusionRequestState], incoming_eff: float) -> float:
        if not states:
            return 0.0
        current = self._estimate_step_cost(states).step_ms
        if self.cost_model is None:
            current_eff = self._effective_batch_size(states)
            scale = (1.0 + self.batch_growth_alpha * max(current_eff + incoming_eff - 1.0, 0.0)) / (
                1.0 + self.batch_growth_alpha * max(current_eff - 1.0, 0.0)
            )
            return max(current * scale - current, 0.0)
        sampling = states[0].req.sampling_params
        width, height = _sampling_dimensions(sampling)
        after = self.cost_model.estimate(
            model=getattr(self.od_config, "model", None),
            width=width,
            height=height,
            num_frames=getattr(sampling, "num_frames", 1),
            batch_size=len(states) + 1,
            effective_batch_size=self._effective_batch_size(states) + incoming_eff,
        ).step_ms
        return max(after - current, 0.0)

    def _shape_fallback_step_ms(self, state: DiffusionRequestState) -> float:
        if not self.use_shape_fallback_cost:
            return self.default_step_ms
        sampling = state.req.sampling_params
        height = _coerce_float(getattr(sampling, "height", None))
        width = _coerce_float(getattr(sampling, "width", None))
        if height is None or width is None:
            resolution = _coerce_float(getattr(sampling, "resolution", None))
            height = height or resolution or 1024.0
            width = width or resolution or 1024.0
        frames = _coerce_float(getattr(sampling, "num_frames", None)) or 1.0
        area_scale = max((height * width) / float(1024 * 1024), 0.25)
        return max(self.default_step_ms * area_scale * max(frames, 1.0), 0.001)

    def _effective_batch_size(self, states: list[DiffusionRequestState]) -> float:
        return sum(
            estimate_request_effective_size(
                state.req.sampling_params,
                prompts=getattr(state.req, "prompts", None),
            )
            for state in states
        )


@dataclass
class _StepSnapshotCost:
    step_ms: float
    source: str
    latent_tokens: int | None


def _get_slo_scheduler_config(od_config: Any) -> dict[str, Any]:
    additional_config = getattr(od_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        return {}
    raw = additional_config.get("diffusion_slo_scheduler") or additional_config.get("slo_scheduler") or {}
    return raw if isinstance(raw, dict) else {}


def _sampling_dimensions(sampling: Any) -> tuple[float | None, float | None]:
    height = _coerce_float(getattr(sampling, "height", None))
    width = _coerce_float(getattr(sampling, "width", None))
    return width, height


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_positive_float(value: Any, default: float) -> float:
    parsed = _coerce_float(value)
    if parsed is None or parsed <= 0:
        return default
    return parsed


def _coerce_nonnegative_float(value: Any, default: float) -> float:
    parsed = _coerce_float(value)
    if parsed is None or parsed < 0:
        return default
    return parsed


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default
