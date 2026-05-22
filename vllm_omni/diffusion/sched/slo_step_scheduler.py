# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionRequestState,
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    NewRequestData,
    SamplingParamsKey,
)
from vllm_omni.diffusion.sched.step_scheduler import StepScheduler

logger = init_logger(__name__)


class SloStepScheduler(StepScheduler):
    """Step-boundary scheduler that chooses one compatible denoise bucket.

    This first-stage policy keeps the existing homogeneous batching contract:
    every scheduled step contains requests with the same SamplingParamsKey.
    The difference from StepScheduler is that WAITING no longer has strict
    FIFO head-of-line blocking, and RUNNING buckets can be skipped at step
    boundaries when another bucket has tighter completion-aware laxity.
    """

    def __init__(self) -> None:
        super().__init__()
        self.default_step_ms = 1.0
        self.batch_growth_alpha = 0.60
        self.decode_ms = 0.0
        self.min_laxity_guard_ms = 0.0

    def initialize(self, od_config) -> None:
        super().initialize(od_config)
        slo_config = _get_slo_scheduler_config(od_config)
        self.default_step_ms = _coerce_positive_float(slo_config.get("default_step_ms"), 1.0)
        self.batch_growth_alpha = _coerce_nonnegative_float(slo_config.get("batch_growth_alpha"), 0.60)
        self.decode_ms = _coerce_nonnegative_float(slo_config.get("decode_ms"), 0.0)
        self.min_laxity_guard_ms = _coerce_float(slo_config.get("min_laxity_guard_ms")) or 0.0

    def schedule(self) -> DiffusionSchedulerOutput:
        if not self._has_any_deadline():
            return super().schedule()

        now_s = time.time()
        bucket = self._select_bucket(now_s)
        if bucket is None:
            return self._empty_schedule()

        selected_key, selected_ids = bucket
        selected_set = set(selected_ids)
        original_running = list(self._running)
        original_running_set = set(original_running)
        preempted_running = [sched_req_id for sched_req_id in original_running if sched_req_id not in selected_set]
        preempted_set = set(preempted_running)

        for sched_req_id in preempted_running:
            state = self._request_states.get(sched_req_id)
            if state is not None and not state.is_finished():
                state.status = DiffusionRequestStatus.PREEMPTED
        self._running = [sched_req_id for sched_req_id in original_running if sched_req_id in selected_set]
        self._waiting = deque(
            [
                *preempted_running,
                *(sid for sid in self._waiting if sid not in selected_set and sid not in preempted_set),
            ]
        )

        scheduled_new_reqs: list[NewRequestData] = []
        scheduled_cached_req_ids: list[str] = []

        for sched_req_id in selected_ids:
            state = self._request_states.get(sched_req_id)
            if state is None or state.is_finished():
                continue
            if sched_req_id in original_running_set:
                scheduled_cached_req_ids.append(sched_req_id)
                continue

            was_new_request = state.status == DiffusionRequestStatus.WAITING
            state.status = DiffusionRequestStatus.RUNNING
            if sched_req_id not in self._running:
                self._running.append(sched_req_id)
            if was_new_request:
                scheduled_new_reqs.append(NewRequestData.from_state(state))
            else:
                scheduled_cached_req_ids.append(sched_req_id)

        self._running_sampling_params_key = selected_key if self._running else None

        scheduler_output = DiffusionSchedulerOutput(
            step_id=self._step_id,
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=CachedRequestData(sched_req_ids=scheduled_cached_req_ids),
            finished_req_ids=set(self._finished_req_ids),
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._waiting),
        )
        self._step_id += 1
        self._finished_req_ids.clear()
        return scheduler_output

    def get_load_snapshot(self) -> dict[str, Any]:
        now_s = time.time()
        buckets = []
        for key, states in self._candidate_batches(now_s):
            step_ms = self._estimate_step_ms(states)
            buckets.append(
                {
                    "key": None if key is None else asdict(key),
                    "num_running": sum(1 for state in states if state.sched_req_id in self._running),
                    "num_waiting": sum(1 for state in states if state.sched_req_id in self._waiting),
                    "candidate_batch_size": len(states),
                    "effective_batch_size": self._effective_batch_size(states),
                    "estimated_step_ms": step_ms,
                    "min_laxity_ms": self._bucket_min_laxity_ms(states, step_ms, now_s),
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

    def _empty_schedule(self) -> DiffusionSchedulerOutput:
        output = DiffusionSchedulerOutput(
            step_id=self._step_id,
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            finished_req_ids=set(self._finished_req_ids),
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._waiting),
        )
        self._step_id += 1
        self._finished_req_ids.clear()
        return output

    def _has_any_deadline(self) -> bool:
        for sched_req_id in [*self._running, *self._waiting]:
            state = self._request_states.get(sched_req_id)
            if state is not None and state.deadline_time_s is not None:
                return True
        return False

    def _select_bucket(self, now_s: float) -> tuple[SamplingParamsKey | None, list[str]] | None:
        candidates: list[tuple[tuple[float, float, float, int], SamplingParamsKey | None, list[str]]] = []
        for key, states in self._candidate_batches(now_s):
            step_ms = self._estimate_step_ms(states)
            min_laxity_ms = self._bucket_min_laxity_ms(states, step_ms, now_s)
            has_deadline = 0.0 if math.isfinite(min_laxity_ms) else 1.0
            oldest_arrival_s = min(state.arrival_time_s for state in states)
            score = (has_deadline, min_laxity_ms, -float(len(states)), int(oldest_arrival_s * 1000))
            candidates.append((score, key, [state.sched_req_id for state in states]))

        if not candidates:
            return None

        all_without_deadline = all(candidate[0][0] > 0 for candidate in candidates)
        if all_without_deadline:
            return self._fifo_fallback_bucket()

        _, key, sched_req_ids = min(candidates, key=lambda item: item[0])
        return key, sched_req_ids

    def _fifo_fallback_bucket(self) -> tuple[SamplingParamsKey | None, list[str]] | None:
        batches = self._candidate_batches(time.time())
        if not batches:
            return None
        key, states = batches[0]
        return key, [state.sched_req_id for state in states]

    def _candidate_batches(self, now_s: float) -> list[tuple[SamplingParamsKey | None, list[DiffusionRequestState]]]:
        grouped: dict[SamplingParamsKey, list[DiffusionRequestState]] = {}
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
                grouped.setdefault(key, []).append(state)

        buckets: list[tuple[SamplingParamsKey | None, list[DiffusionRequestState]]] = []
        for key, states in grouped.items():
            selected = self._select_states_for_key(states, now_s)
            if selected:
                buckets.append((key, selected))
        buckets.extend(singleton_buckets)
        return buckets

    def _select_states_for_key(self, states: list[DiffusionRequestState], now_s: float) -> list[DiffusionRequestState]:
        ordered = sorted(states, key=lambda state: self._state_admission_priority(state, now_s))
        selected: list[DiffusionRequestState] = []
        for state in ordered[: self.max_num_running_reqs]:
            candidate = [*selected, state]
            if selected and self._has_deadline(candidate):
                step_ms = self._estimate_step_ms(candidate)
                if self._bucket_min_laxity_ms(candidate, step_ms, now_s) < self.min_laxity_guard_ms:
                    continue
            selected.append(state)
        if not selected and ordered:
            selected.append(ordered[0])
        return selected

    def _state_admission_priority(self, state: DiffusionRequestState, now_s: float) -> tuple[float, float, float]:
        step_ms = self._estimate_step_ms([state])
        laxity_ms = self._bucket_min_laxity_ms([state], step_ms, now_s)
        has_deadline = 0.0 if math.isfinite(laxity_ms) else 1.0
        return (has_deadline, laxity_ms, state.arrival_time_s)

    @staticmethod
    def _has_deadline(states: list[DiffusionRequestState]) -> bool:
        return any(state.deadline_time_s is not None for state in states)

    def _bucket_min_laxity_ms(self, states: list[DiffusionRequestState], step_ms: float, now_s: float) -> float:
        laxities: list[float] = []
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

    def _estimate_step_ms(self, states: list[DiffusionRequestState]) -> float:
        single_step_candidates: list[float] = []
        for state in states:
            explicit = _coerce_float(_extra_arg(state, "estimated_step_ms"))
            if explicit is not None and explicit > 0:
                single_step_candidates.append(explicit)
                continue

            progress = self._request_progress.get(state.sched_req_id)
            if state.reference_cost_ms is not None and state.reference_cost_ms > 0 and progress is not None:
                single_step_candidates.append(max(state.reference_cost_ms / max(progress.total_steps, 1), 0.001))
                continue

            single_step_candidates.append(self._shape_fallback_step_ms(state))

        single_step_ms = max(single_step_candidates) if single_step_candidates else self.default_step_ms
        b_eff = self._effective_batch_size(states)
        batch_scale = 1.0 + self.batch_growth_alpha * max(b_eff - 1.0, 0.0)
        return max(single_step_ms * batch_scale, 0.001)

    def _shape_fallback_step_ms(self, state: DiffusionRequestState) -> float:
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
        return sum(_request_effective_size(state) for state in states)


def _request_effective_size(state: DiffusionRequestState) -> float:
    sampling = state.req.sampling_params
    outputs = _coerce_float(getattr(sampling, "num_outputs_per_prompt", None)) or 1.0
    guidance = 1.0
    true_cfg_scale = _coerce_float(getattr(sampling, "true_cfg_scale", None))
    guidance_scale = _coerce_float(getattr(sampling, "guidance_scale", None))
    if bool(getattr(sampling, "do_classifier_free_guidance", False)) or true_cfg_scale is not None:
        guidance = 2.0
    elif guidance_scale is not None and guidance_scale > 1.0:
        guidance = 2.0
    return max(outputs, 1.0) * guidance


def _get_slo_scheduler_config(od_config: Any) -> dict[str, Any]:
    additional_config = getattr(od_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        return {}
    raw = additional_config.get("diffusion_slo_scheduler") or additional_config.get("slo_scheduler") or {}
    return raw if isinstance(raw, dict) else {}


def _extra_arg(state: DiffusionRequestState, key: str) -> Any:
    extra_args = getattr(state.req.sampling_params, "extra_args", None)
    if isinstance(extra_args, dict):
        return extra_args.get(key)
    return None


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
