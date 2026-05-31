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
from vllm_omni.diffusion.sched.step_cost_model import (
    DiffusionStepCostModel,
    estimate_request_effective_size,
)

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
        self.cost_model: DiffusionStepCostModel | None = None
        self.preemption_laxity_margin_ms = 0.0
        self.preemption_safe_laxity_ms = 1000.0
        self.enable_step_preemption = True
        self.no_preemption_admission_guard = True
        self.no_preemption_admission_max_batch_size = 0
        self.ignore_request_reference_cost = False
        self.use_shape_fallback_cost = True
        self.pard_oracle_enabled = False
        self.pard_oracle_snapshot_enabled = False
        self.pard_oracle_laxity_margin_ms = 0.0
        self.pard_oracle_max_records = 16
        self.pard_drop_mode = "off"
        self.pard_drop_laxity_margin_ms = 0.0
        self.pard_drop_max_per_schedule = 0

    def initialize(self, od_config) -> None:
        super().initialize(od_config)
        slo_config = _get_slo_scheduler_config(od_config)
        self.default_step_ms = _coerce_positive_float(slo_config.get("default_step_ms"), 1.0)
        self.batch_growth_alpha = _coerce_nonnegative_float(slo_config.get("batch_growth_alpha"), 0.60)
        self.decode_ms = _coerce_nonnegative_float(slo_config.get("decode_ms"), 0.0)
        self.min_laxity_guard_ms = _coerce_float(slo_config.get("min_laxity_guard_ms")) or 0.0
        self.preemption_laxity_margin_ms = _coerce_nonnegative_float(
            slo_config.get("preemption_laxity_margin_ms"),
            0.0,
        )
        self.preemption_safe_laxity_ms = _coerce_nonnegative_float(
            slo_config.get("preemption_safe_laxity_ms"),
            1000.0,
        )
        self.enable_step_preemption = _coerce_bool(slo_config.get("enable_step_preemption"), True)
        self.no_preemption_admission_guard = _coerce_bool(
            slo_config.get("no_preemption_admission_guard"),
            True,
        )
        self.no_preemption_admission_max_batch_size = int(
            _coerce_nonnegative_float(
                slo_config.get("no_preemption_admission_max_batch_size"),
                0.0,
            )
        )
        self.ignore_request_reference_cost = _coerce_bool(slo_config.get("ignore_request_reference_cost"), False)
        self.use_shape_fallback_cost = _coerce_bool(slo_config.get("use_shape_fallback_cost"), True)
        self.pard_oracle_enabled = _coerce_bool(slo_config.get("pard_oracle_enabled"), False)
        self.pard_oracle_snapshot_enabled = _coerce_bool(
            slo_config.get("pard_oracle_snapshot_enabled"),
            False,
        )
        self.pard_oracle_laxity_margin_ms = _coerce_float(
            slo_config.get("pard_oracle_laxity_margin_ms")
        ) or 0.0
        self.pard_oracle_max_records = int(
            _coerce_nonnegative_float(slo_config.get("pard_oracle_max_records"), 16.0)
        )
        self.pard_drop_mode = _coerce_pard_drop_mode(slo_config.get("pard_drop_mode"))
        self.pard_drop_laxity_margin_ms = _coerce_float(
            slo_config.get("pard_drop_laxity_margin_ms")
        ) or 0.0
        self.pard_drop_max_per_schedule = int(
            _coerce_nonnegative_float(slo_config.get("pard_drop_max_per_schedule"), 0.0)
        )
        self.cost_model = DiffusionStepCostModel.from_config(
            slo_config,
            default_step_ms=self.default_step_ms,
        )

    def schedule(self) -> DiffusionSchedulerOutput:
        if not self._has_any_deadline():
            return super().schedule()

        now_s = time.time()
        pard_drop_debug = self._apply_pard_drops(now_s)
        if pard_drop_debug is not None and not self._has_any_deadline():
            return super().schedule()
        bucket = self._select_bucket(now_s)
        if bucket is None:
            return self._empty_schedule()

        selected_key, selected_ids, debug_info = bucket
        pard_oracle_debug = (
            self._pard_oracle_debug(now_s, selected_ids) if self.pard_oracle_enabled else None
        )
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

        debug_info = dict(debug_info)
        debug_info["resident_reqs"] = len(self._resident_sched_req_ids())
        debug_info["safe_admit_capacity"] = self._safe_admit_capacity()
        if pard_oracle_debug is not None:
            debug_info["pard_oracle"] = pard_oracle_debug
        if pard_drop_debug is not None:
            debug_info["pard_drop"] = pard_drop_debug

        scheduler_output = DiffusionSchedulerOutput(
            step_id=self._step_id,
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=CachedRequestData(sched_req_ids=scheduled_cached_req_ids),
            finished_req_ids=set(self._finished_req_ids),
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._waiting),
            debug_info=debug_info,
        )
        self._step_id += 1
        self._finished_req_ids.clear()
        return scheduler_output

    def get_load_snapshot(self) -> dict[str, Any]:
        now_s = time.time()
        buckets = []
        for key, states in self._candidate_batches(now_s):
            step_estimate = self._estimate_step_cost(states)
            step_ms = step_estimate.step_ms
            delta_one_ms = self._estimate_incremental_step_ms(states, 1.0)
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
                    "estimated_step_ms": step_ms,
                    "estimated_step_ms_if_add_one": step_ms + delta_one_ms,
                    "incremental_step_ms_if_add_one": delta_one_ms,
                    "step_cost_source": step_estimate.source,
                    "latent_tokens": step_estimate.latent_tokens,
                    "shape": {
                        "width": width,
                        "height": height,
                        "num_frames": frames,
                    },
                    "min_laxity_ms": _json_float(self._bucket_min_laxity_ms(states, step_ms, now_s)),
                    "min_remaining_steps": min(self._remaining_steps(state) for state in states),
                    "max_remaining_steps": max(self._remaining_steps(state) for state in states),
                    "oldest_arrival_time_s": min(state.arrival_time_s for state in states),
                }
            )
        snapshot = {
            "policy": self.__class__.__name__,
            "timestamp_s": now_s,
            "num_waiting": len(self._waiting),
            "num_running": len(self._running),
            "num_resident_reqs": len(self._resident_sched_req_ids()),
            "max_num_running": self.max_num_running_reqs,
            "safe_admit_capacity": self._safe_admit_capacity(),
            "no_preemption_admission_max_batch_size": self.no_preemption_admission_max_batch_size,
            "buckets": buckets,
        }
        if self.pard_oracle_enabled and self.pard_oracle_snapshot_enabled:
            snapshot["pard_oracle"] = self._pard_oracle_debug(now_s, [])
        return snapshot

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

    def _select_bucket(self, now_s: float) -> tuple[SamplingParamsKey | None, list[str], dict[str, Any]] | None:
        candidates: list[tuple[tuple[float, float, float, float, float, int], SamplingParamsKey | None, list[str]]] = []
        for key, states in self._candidate_batches(now_s):
            step_ms = self._estimate_step_ms(states)
            min_laxity_ms = self._bucket_min_laxity_ms(states, step_ms, now_s)
            min_laxity_ratio = self._bucket_min_laxity_ratio(states, step_ms, now_s)
            has_deadline = 0.0 if math.isfinite(min_laxity_ms) else 1.0
            guard_rank = 0.0 if min_laxity_ms < self.min_laxity_guard_ms else 1.0
            oldest_arrival_s = min(state.arrival_time_s for state in states)
            ratio_score = min_laxity_ratio if guard_rank > 0 else min_laxity_ms
            score = (has_deadline, guard_rank, ratio_score, min_laxity_ms, -float(len(states)), int(oldest_arrival_s * 1000))
            candidates.append((score, key, [state.sched_req_id for state in states]))

        if not candidates:
            return None

        all_without_deadline = all(candidate[0][0] > 0 for candidate in candidates)
        if all_without_deadline:
            fallback = self._fifo_fallback_bucket()
            if fallback is None:
                return None
            key, sched_req_ids = fallback
            return key, sched_req_ids, {"policy": "fifo_fallback", "candidate_count": len(candidates)}

        selected = min(candidates, key=lambda item: item[0])
        current = self._current_running_candidate(candidates)
        if current is not None and current is not selected:
            if not self.enable_step_preemption or self._should_keep_current_bucket(current, selected):
                selected = current

        _, key, sched_req_ids = selected
        return key, sched_req_ids, self._selection_debug(candidates, selected, now_s)

    def _selection_debug(
        self,
        candidates: list[tuple[tuple[float, float, float, float, float, int], SamplingParamsKey | None, list[str]]],
        selected: tuple[tuple[float, float, float, float, float, int], SamplingParamsKey | None, list[str]],
        now_s: float,
    ) -> dict[str, Any]:
        debug_candidates = []
        for score, key, sched_req_ids in candidates:
            debug_candidates.append(
                {
                    "key": None if key is None else asdict(key),
                    "sched_req_ids": list(sched_req_ids),
                    "score": [_json_float(value) for value in score],
                    "min_laxity_ms": _json_float(score[3]),
                    "batch_size": len(sched_req_ids),
                }
            )
        _, selected_key, selected_ids = selected
        return {
            "policy": "slo",
            "timestamp_s": now_s,
            "enable_step_preemption": self.enable_step_preemption,
            "selected_key": None if selected_key is None else asdict(selected_key),
            "selected_req_ids": list(selected_ids),
            "candidate_count": len(candidates),
            "candidates": debug_candidates,
        }

    def _current_running_candidate(
        self,
        candidates: list[tuple[tuple[float, float, float, float, float, int], SamplingParamsKey | None, list[str]]],
    ) -> tuple[tuple[float, float, float, float, float, int], SamplingParamsKey | None, list[str]] | None:
        if not self._running:
            return None
        running_set = set(self._running)
        for candidate in candidates:
            _, _, sched_req_ids = candidate
            if running_set.intersection(sched_req_ids):
                return candidate
        return None

    def _should_keep_current_bucket(
        self,
        current: tuple[tuple[float, float, float, float, float, int], SamplingParamsKey | None, list[str]],
        selected: tuple[tuple[float, float, float, float, float, int], SamplingParamsKey | None, list[str]],
    ) -> bool:
        current_laxity_ms = current[0][3]
        selected_laxity_ms = selected[0][3]
        if not (math.isfinite(current_laxity_ms) and math.isfinite(selected_laxity_ms)):
            return False
        if selected_laxity_ms <= self.preemption_safe_laxity_ms:
            return False
        if current_laxity_ms >= self.min_laxity_guard_ms and selected_laxity_ms >= self.min_laxity_guard_ms:
            return True
        return selected_laxity_ms >= current_laxity_ms - self.preemption_laxity_margin_ms

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
        resident_ids = self._resident_sched_req_ids()

        for sched_req_id in [*self._running, *self._waiting]:
            if sched_req_id in seen_req_ids:
                continue
            seen_req_ids.add(sched_req_id)
            state = self._request_states.get(sched_req_id)
            if state is None or state.is_finished():
                continue
            key = state.sampling_params_key
            if key is None:
                if self._can_admit_state(state, resident_ids, 0):
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
        resident_ids = self._resident_sched_req_ids()
        if not self.enable_step_preemption:
            running = [state for state in states if state.sched_req_id in self._running]
            waiting = [state for state in states if state.sched_req_id not in self._running]
            selected = running[: self.max_num_running_reqs]
            if len(selected) >= self.max_num_running_reqs:
                return selected
            new_admissions = sum(1 for state in selected if state.sched_req_id not in resident_ids)
            waiting = sorted(waiting, key=lambda state: self._state_admission_priority(state, now_s))
            return self._append_admissible_states(
                selected,
                waiting,
                resident_ids,
                new_admissions,
                now_s,
                protect_laxity=self.no_preemption_admission_guard,
            )

        ordered = sorted(states, key=lambda state: self._state_admission_priority(state, now_s))
        selected: list[DiffusionRequestState] = []
        new_admissions = 0
        for state in ordered:
            if len(selected) >= self.max_num_running_reqs:
                break
            if not self._can_admit_state(state, resident_ids, new_admissions):
                continue
            if self._would_violate_admission_guard(selected, state, now_s):
                continue
            selected.append(state)
            if state.sched_req_id not in resident_ids:
                new_admissions += 1
        return selected

    def _append_admissible_states(
        self,
        selected: list[DiffusionRequestState],
        waiting: list[DiffusionRequestState],
        resident_ids: set[str],
        new_admissions: int,
        now_s: float,
        *,
        protect_laxity: bool = False,
    ) -> list[DiffusionRequestState]:
        for state in waiting:
            if len(selected) >= self.max_num_running_reqs:
                break
            if (
                protect_laxity
                and self.no_preemption_admission_max_batch_size > 0
                and len(selected) >= self.no_preemption_admission_max_batch_size
            ):
                break
            if not self._can_admit_state(state, resident_ids, new_admissions):
                continue
            if protect_laxity and self._would_violate_admission_guard(selected, state, now_s):
                continue
            selected.append(state)
            if state.sched_req_id not in resident_ids:
                new_admissions += 1
        return selected

    def _would_violate_admission_guard(
        self,
        selected: list[DiffusionRequestState],
        state: DiffusionRequestState,
        now_s: float,
    ) -> bool:
        if not selected:
            return False
        candidate = [*selected, state]
        if not self._has_deadline(candidate):
            return False

        step_ms = self._estimate_step_ms(candidate)
        candidate_laxity_ms = self._bucket_min_laxity_ms(candidate, step_ms, now_s)
        selected_step_ms = self._estimate_step_ms(selected)
        selected_laxity_ms = self._bucket_min_laxity_ms(selected, selected_step_ms, now_s)
        if (
            selected_laxity_ms >= 0.0
            and candidate_laxity_ms < self.min_laxity_guard_ms
            and candidate_laxity_ms < selected_laxity_ms
        ):
            return True

        state_single_laxity_ms = self._state_laxity_ms(
            state,
            self._estimate_step_ms([state]),
            now_s,
        )
        state_candidate_laxity_ms = self._state_laxity_ms(state, step_ms, now_s)
        return (
            state_single_laxity_ms >= 0.0
            and state_candidate_laxity_ms < self.min_laxity_guard_ms
            and state_candidate_laxity_ms < state_single_laxity_ms
        )

    def _resident_sched_req_ids(self) -> set[str]:
        resident_ids = set(self._running)
        for sched_req_id, state in self._request_states.items():
            if state.is_finished():
                continue
            if state.status in (DiffusionRequestStatus.RUNNING, DiffusionRequestStatus.PREEMPTED):
                resident_ids.add(sched_req_id)
        return resident_ids

    def _safe_admit_capacity(self) -> int:
        return max(0, self.max_num_running_reqs - len(self._resident_sched_req_ids()))

    def _can_admit_state(
        self,
        state: DiffusionRequestState,
        resident_ids: set[str],
        new_admissions: int,
    ) -> bool:
        if state.sched_req_id in resident_ids:
            return True
        return len(resident_ids) + new_admissions < self.max_num_running_reqs

    def _state_admission_priority(self, state: DiffusionRequestState, now_s: float) -> tuple[float, float, float]:
        step_ms = self._estimate_step_ms([state])
        laxity_ms = self._bucket_min_laxity_ms([state], step_ms, now_s)
        has_deadline = 0.0 if math.isfinite(laxity_ms) else 1.0
        return (has_deadline, laxity_ms, state.arrival_time_s)

    def _apply_pard_drops(self, now_s: float) -> dict[str, Any] | None:
        if self.pard_drop_mode == "off":
            return None

        resident_ids = self._resident_sched_req_ids()
        drop_ids: list[str] = []
        records: list[dict[str, Any]] = []
        for sched_req_id in [*self._waiting, *self._running]:
            if self.pard_drop_max_per_schedule > 0 and len(drop_ids) >= self.pard_drop_max_per_schedule:
                break
            state = self._request_states.get(sched_req_id)
            if state is None or state.is_finished() or state.deadline_time_s is None:
                continue

            is_resident = sched_req_id in resident_ids
            if self.pard_drop_mode == "admission_only" and is_resident:
                continue

            single_step_ms = self._estimate_step_ms([state])
            optimistic_laxity_ms = self._state_laxity_ms(state, single_step_ms, now_s)
            if (
                not math.isfinite(optimistic_laxity_ms)
                or optimistic_laxity_ms >= self.pard_drop_laxity_margin_ms
            ):
                continue

            drop_ids.append(sched_req_id)
            progress = self._request_progress.get(sched_req_id)
            records.append(
                {
                    "sched_req_id": sched_req_id,
                    "status": state.status.name,
                    "resident": is_resident,
                    "remaining_steps": self._remaining_steps(state),
                    "current_step": None if progress is None else progress.current_step,
                    "total_steps": None if progress is None else progress.total_steps,
                    "single_step_ms": single_step_ms,
                    "optimistic_laxity_ms": _json_float(optimistic_laxity_ms),
                    "deadline_time_s": state.deadline_time_s,
                    "time_to_deadline_ms": _json_float((state.deadline_time_s - now_s) * 1000.0),
                }
            )

        if drop_ids:
            self._finish_requests(
                {sched_req_id: DiffusionRequestStatus.FINISHED_ABORTED for sched_req_id in drop_ids}
            )

        return {
            "enabled": True,
            "mode": self.pard_drop_mode,
            "laxity_margin_ms": self.pard_drop_laxity_margin_ms,
            "max_per_schedule": self.pard_drop_max_per_schedule,
            "dropped_req_count": len(drop_ids),
            "dropped_req_ids": drop_ids[: self.pard_oracle_max_records],
            "records": records[: self.pard_oracle_max_records],
        }

    def _pard_oracle_debug(self, now_s: float, selected_req_ids: list[str]) -> dict[str, Any]:
        selected_set = set(selected_req_ids)
        selected_states = [
            state
            for sched_req_id in selected_req_ids
            if (state := self._request_states.get(sched_req_id)) is not None and not state.is_finished()
        ]
        selected_step_ms = self._estimate_step_ms(selected_states) if selected_states else None

        bucket_step_by_req: dict[str, float] = {}
        bucket_ids_by_req: dict[str, list[str]] = {}
        candidate_buckets = []
        for key, states in self._candidate_batches(now_s):
            if not states:
                continue
            step_ms = self._estimate_step_ms(states)
            sched_req_ids = [state.sched_req_id for state in states]
            for state in states:
                bucket_step_by_req[state.sched_req_id] = step_ms
                bucket_ids_by_req[state.sched_req_id] = sched_req_ids
            candidate_buckets.append(
                {
                    "key": None if key is None else asdict(key),
                    "sched_req_ids": sched_req_ids,
                    "estimated_step_ms": step_ms,
                    "min_laxity_ms": _json_float(self._bucket_min_laxity_ms(states, step_ms, now_s)),
                    "batch_size": len(states),
                }
            )

        resident_ids = self._resident_sched_req_ids()
        records: list[dict[str, Any]] = []
        admission_would_drop_req_ids: list[str] = []
        step_would_drop_req_ids: list[str] = []
        projected_miss_req_ids: list[str] = []
        seen_req_ids: set[str] = set()
        for sched_req_id in [*self._running, *self._waiting]:
            if sched_req_id in seen_req_ids:
                continue
            seen_req_ids.add(sched_req_id)
            state = self._request_states.get(sched_req_id)
            if state is None or state.is_finished():
                continue

            single_step_ms = self._estimate_step_ms([state])
            if selected_step_ms is not None and sched_req_id in selected_set:
                projected_step_ms = selected_step_ms
                projected_step_source = "selected_bucket"
            elif sched_req_id in bucket_step_by_req:
                projected_step_ms = bucket_step_by_req[sched_req_id]
                projected_step_source = "candidate_bucket"
            else:
                projected_step_ms = single_step_ms
                projected_step_source = "single_request"

            optimistic_laxity_ms = self._state_laxity_ms(state, single_step_ms, now_s)
            projected_laxity_ms = self._state_laxity_ms(state, projected_step_ms, now_s)
            optimistic_miss = (
                math.isfinite(optimistic_laxity_ms)
                and optimistic_laxity_ms < self.pard_oracle_laxity_margin_ms
            )
            projected_miss = (
                math.isfinite(projected_laxity_ms)
                and projected_laxity_ms < self.pard_oracle_laxity_margin_ms
            )
            is_resident = sched_req_id in resident_ids
            would_drop_admission = (not is_resident) and optimistic_miss
            would_drop_step = is_resident and optimistic_miss
            if would_drop_admission:
                admission_would_drop_req_ids.append(sched_req_id)
            if would_drop_step:
                step_would_drop_req_ids.append(sched_req_id)
            if projected_miss:
                projected_miss_req_ids.append(sched_req_id)

            progress = self._request_progress.get(sched_req_id)
            records.append(
                {
                    "sched_req_id": sched_req_id,
                    "status": state.status.name,
                    "resident": is_resident,
                    "in_selected_bucket": sched_req_id in selected_set,
                    "candidate_bucket_req_ids": bucket_ids_by_req.get(sched_req_id, [sched_req_id]),
                    "current_step": None if progress is None else progress.current_step,
                    "total_steps": None if progress is None else progress.total_steps,
                    "remaining_steps": self._remaining_steps(state),
                    "single_step_ms": single_step_ms,
                    "projected_step_ms": projected_step_ms,
                    "projected_step_source": projected_step_source,
                    "deadline_time_s": state.deadline_time_s,
                    "time_to_deadline_ms": _json_float(
                        (state.deadline_time_s - now_s) * 1000.0
                        if state.deadline_time_s is not None
                        else math.inf
                    ),
                    "optimistic_laxity_ms": _json_float(optimistic_laxity_ms),
                    "projected_laxity_ms": _json_float(projected_laxity_ms),
                    "optimistic_miss": optimistic_miss,
                    "projected_miss": projected_miss,
                    "would_drop_admission": would_drop_admission,
                    "would_drop_step": would_drop_step,
                }
            )

        records.sort(
            key=lambda record: (
                not (record["would_drop_admission"] or record["would_drop_step"]),
                not record["projected_miss"],
                _sort_laxity(record.get("optimistic_laxity_ms")),
                str(record["sched_req_id"]),
            )
        )
        omitted_records = max(0, len(records) - self.pard_oracle_max_records)
        records = records[: self.pard_oracle_max_records]
        would_drop_req_ids = [*admission_would_drop_req_ids, *step_would_drop_req_ids]

        return {
            "enabled": True,
            "dry_run_only": True,
            "phase": "step_boundary",
            "laxity_margin_ms": self.pard_oracle_laxity_margin_ms,
            "selected_req_ids": list(selected_req_ids),
            "admission_would_drop_req_count": len(admission_would_drop_req_ids),
            "step_would_drop_req_count": len(step_would_drop_req_ids),
            "projected_miss_req_count": len(projected_miss_req_ids),
            "would_drop_req_count": len(would_drop_req_ids),
            "admission_would_drop_req_ids": admission_would_drop_req_ids[: self.pard_oracle_max_records],
            "step_would_drop_req_ids": step_would_drop_req_ids[: self.pard_oracle_max_records],
            "projected_miss_req_ids": projected_miss_req_ids[: self.pard_oracle_max_records],
            "would_drop_req_ids": would_drop_req_ids[: self.pard_oracle_max_records],
            "num_active_reqs": len(seen_req_ids),
            "num_omitted_records": omitted_records,
            "records": records,
            "candidate_buckets": candidate_buckets[:8],
        }

    @staticmethod
    def _has_deadline(states: list[DiffusionRequestState]) -> bool:
        return any(state.deadline_time_s is not None for state in states)

    def _bucket_min_laxity_ms(self, states: list[DiffusionRequestState], step_ms: float, now_s: float) -> float:
        laxities: list[float] = []
        for state in states:
            if state.deadline_time_s is None:
                continue
            laxities.append(self._state_laxity_ms(state, step_ms, now_s))
        return min(laxities) if laxities else math.inf

    def _state_laxity_ms(self, state: DiffusionRequestState, step_ms: float, now_s: float) -> float:
        if state.deadline_time_s is None:
            return math.inf
        remaining_work_ms = self._remaining_steps(state) * step_ms + self.decode_ms
        return (state.deadline_time_s - now_s) * 1000.0 - remaining_work_ms

    def _bucket_min_laxity_ratio(self, states: list[DiffusionRequestState], step_ms: float, now_s: float) -> float:
        ratios: list[float] = []
        for state in states:
            if state.deadline_time_s is None:
                continue
            remaining_work_ms = self._remaining_steps(state) * step_ms + self.decode_ms
            laxity_ms = (state.deadline_time_s - now_s) * 1000.0 - remaining_work_ms
            ratios.append(laxity_ms / max(remaining_work_ms, 0.001))
        return min(ratios) if ratios else math.inf

    def _remaining_steps(self, state: DiffusionRequestState) -> int:
        progress = self._request_progress.get(state.sched_req_id)
        if progress is None:
            return 1
        return max(progress.total_steps - progress.current_step, 1)

    def _estimate_step_ms(self, states: list[DiffusionRequestState]) -> float:
        return self._estimate_step_cost(states).step_ms

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
            explicit = _coerce_float(_extra_arg(state, "estimated_step_ms"))
            if explicit is not None and explicit > 0:
                single_step_candidates.append(explicit)
                continue

            progress = self._request_progress.get(state.sched_req_id)
            if (
                not self.ignore_request_reference_cost
                and state.reference_cost_ms is not None
                and state.reference_cost_ms > 0
                and progress is not None
            ):
                single_step_candidates.append(max(state.reference_cost_ms / max(progress.total_steps, 1), 0.001))
                continue

            single_step_candidates.append(self._shape_fallback_step_ms(state))

        single_step_ms = max(single_step_candidates) if single_step_candidates else self.default_step_ms
        b_eff = self._effective_batch_size(states)
        batch_scale = 1.0 + self.batch_growth_alpha * max(b_eff - 1.0, 0.0)
        step_ms = max(single_step_ms * batch_scale, 0.001)
        return _LegacyStepCostEstimate(step_ms, "legacy", None)

    def _estimate_incremental_step_ms(self, states: list[DiffusionRequestState], incoming_eff: float) -> float:
        if not states:
            return 0.0
        current = self._estimate_step_ms(states)
        if self.cost_model is None:
            if current <= 0:
                return 0.0
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
        return sum(_request_effective_size(state) for state in states)


def _request_effective_size(state: DiffusionRequestState) -> float:
    return _request_prompt_count(state) * estimate_request_effective_size(
        state.req.sampling_params,
        prompts=getattr(state.req, "prompts", None),
    )


def _request_prompt_count(state: DiffusionRequestState) -> float:
    prompts = getattr(state.req, "prompts", None)
    if isinstance(prompts, (list, tuple)):
        return float(max(len(prompts), 1))
    return 1.0


class _LegacyStepCostEstimate:
    def __init__(self, step_ms: float, source: str, latent_tokens: int | None) -> None:
        self.step_ms = step_ms
        self.source = source
        self.latent_tokens = latent_tokens


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


def _coerce_pard_drop_mode(value: Any) -> str:
    if value is None:
        return "off"
    mode = str(value).strip().lower()
    if mode in {"admission_only", "step_boundary"}:
        return mode
    return "off"


def _json_float(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _sort_laxity(value: Any) -> float:
    parsed = _coerce_float(value)
    return parsed if parsed is not None else math.inf
