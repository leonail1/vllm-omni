# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from vllm_omni.diffusion.sched.interface import DiffusionRequestState
from vllm_omni.diffusion.sched.slo_step_scheduler import (
    SloStepScheduler,
    _coerce_bool,
    _coerce_float,
    _extra_arg,
    _request_effective_size,
    _sampling_dimensions,
)


@dataclass(frozen=True)
class _TokenBatchStats:
    max_latent_tokens: int
    total_latent_tokens: int
    token_effective_batch_size: float
    token_batch_utilization: float
    total_token_work: float
    max_token_work: float


@dataclass(frozen=True)
class _TokenStepCostEstimate:
    step_ms: float
    source: str
    latent_tokens: int | None


class TokenSloStepScheduler(SloStepScheduler):
    """SLO scheduler for PR4024 heterogeneous token batches.

    PR4024 allows Qwen-Image requests with different spatial sizes to share a
    denoise step batch. The shape-bucket SLO scheduler estimates such a batch
    from the first request's shape, which is unsafe when a large-token request
    is mixed behind a small-token request. This scheduler keeps the same
    step-boundary, no-preemption admission model, but estimates batch cost from
    the largest latent-token request and a token-weighted effective batch size.
    """

    def __init__(self) -> None:
        super().__init__()
        self.token_patch_size = 16

    def initialize(self, od_config) -> None:
        super().initialize(od_config)
        slo_config = _get_slo_scheduler_config(od_config)
        self.token_patch_size = max(int(_coerce_float(slo_config.get("latent_patch_size")) or 16), 1)
        self.enable_step_preemption = False
        if "no_preemption_admission_guard" not in slo_config:
            self.no_preemption_admission_guard = True

    def get_load_snapshot(self) -> dict[str, Any]:
        snapshot = super().get_load_snapshot()
        snapshot["policy"] = self.__class__.__name__
        total_pressure = 0.0
        for bucket in snapshot.get("buckets", []):
            if not isinstance(bucket, dict):
                continue
            states = self._states_from_snapshot_ids(bucket.get("sched_req_ids"))
            if not states:
                states = self._states_matching_bucket(bucket.get("key"))
            if not states:
                continue
            stats = self._token_batch_stats(states)
            bucket.update(
                {
                    "max_latent_tokens": stats.max_latent_tokens,
                    "total_latent_tokens": stats.total_latent_tokens,
                    "token_effective_batch_size": stats.token_effective_batch_size,
                    "token_batch_utilization": stats.token_batch_utilization,
                    "total_token_work": stats.total_token_work,
                    "max_token_work": stats.max_token_work,
                }
            )
            representative = max(states, key=self._state_latent_tokens)
            sampling = representative.req.sampling_params
            width, height = _sampling_dimensions(sampling)
            bucket["shape"] = {
                "width": width,
                "height": height,
                "num_frames": _coerce_float(getattr(sampling, "num_frames", None)) or 1.0,
            }
            total_pressure += stats.total_token_work * max(float(bucket.get("max_remaining_steps", 1.0) or 1.0), 1.0)
        snapshot["token_pressure"] = total_pressure
        return snapshot

    def _selection_debug(self, candidates, selected, now_s: float) -> dict[str, Any]:
        debug = super()._selection_debug(candidates, selected, now_s)
        for candidate in debug.get("candidates", []):
            req_ids = candidate.get("sched_req_ids") or []
            states = [self._request_states[req_id] for req_id in req_ids if req_id in self._request_states]
            if states:
                stats = self._token_batch_stats(states)
                candidate.update(
                    {
                        "max_latent_tokens": stats.max_latent_tokens,
                        "total_latent_tokens": stats.total_latent_tokens,
                        "token_effective_batch_size": stats.token_effective_batch_size,
                        "token_batch_utilization": stats.token_batch_utilization,
                    }
                )
        debug["policy"] = "token_slo"
        return debug

    def _state_admission_priority(self, state: DiffusionRequestState, now_s: float) -> tuple[float, float, float, float]:
        base = super()._state_admission_priority(state, now_s)
        return (*base, -self._state_token_work(state))

    def _effective_batch_size(self, states: list[DiffusionRequestState]) -> float:
        if not states:
            return 0.0
        return self._token_batch_stats(states).token_effective_batch_size

    def _estimate_step_cost(self, states: list[DiffusionRequestState]):
        if states and self.cost_model is not None:
            representative = max(states, key=self._state_latent_tokens)
            sampling = representative.req.sampling_params
            width, height = _sampling_dimensions(sampling)
            stats = self._token_batch_stats(states)
            estimate = self.cost_model.estimate(
                model=getattr(self.od_config, "model", None),
                width=width,
                height=height,
                num_frames=getattr(sampling, "num_frames", 1),
                batch_size=len(states),
                effective_batch_size=stats.token_effective_batch_size,
            )
            if estimate.step_ms > 0 and estimate.source != "default":
                return _TokenStepCostEstimate(estimate.step_ms, f"token_{estimate.source}", stats.max_latent_tokens)

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

        stats = self._token_batch_stats(states) if states else None
        single_step_ms = max(single_step_candidates) if single_step_candidates else self.default_step_ms
        b_eff = stats.token_effective_batch_size if stats is not None else 1.0
        batch_scale = 1.0 + self.batch_growth_alpha * max(b_eff - 1.0, 0.0)
        step_ms = max(single_step_ms * batch_scale, 0.001)
        return _TokenStepCostEstimate(
            step_ms,
            "token_legacy",
                None if stats is None else stats.max_latent_tokens,
        )

    def _estimate_incremental_step_ms(self, states: list[DiffusionRequestState], incoming_eff: float) -> float:
        if not states:
            return 0.0
        current = self._estimate_step_ms(states)
        stats = self._token_batch_stats(states)
        incoming_eff = max(float(incoming_eff), 1.0)
        after_eff = stats.token_effective_batch_size + incoming_eff
        if self.cost_model is not None:
            representative = max(states, key=self._state_latent_tokens)
            sampling = representative.req.sampling_params
            width, height = _sampling_dimensions(sampling)
            estimate = self.cost_model.estimate(
                model=getattr(self.od_config, "model", None),
                width=width,
                height=height,
                num_frames=getattr(sampling, "num_frames", 1),
                batch_size=len(states) + 1,
                effective_batch_size=after_eff,
            )
            if estimate.step_ms > 0 and estimate.source != "default":
                return max(estimate.step_ms - current, 0.0)

        if current <= 0:
            return 0.0
        current_eff = stats.token_effective_batch_size
        scale = (1.0 + self.batch_growth_alpha * max(after_eff - 1.0, 0.0)) / (
            1.0 + self.batch_growth_alpha * max(current_eff - 1.0, 0.0)
        )
        return max(current * scale - current, 0.0)

    def _token_batch_stats(self, states: list[DiffusionRequestState]) -> _TokenBatchStats:
        tokens = [self._state_latent_tokens(state) for state in states]
        max_tokens = max(tokens) if tokens else 1
        total_tokens = sum(tokens)
        token_work = [tokens[idx] * _request_effective_size(state) for idx, state in enumerate(states)]
        total_work = sum(token_work)
        max_work = max(token_work) if token_work else float(max_tokens)
        token_eff = max(total_work / max(float(max_tokens), 1.0), 1.0) if states else 0.0
        utilization = total_tokens / max(float(max_tokens * max(len(states), 1)), 1.0)
        return _TokenBatchStats(
            max_latent_tokens=max_tokens,
            total_latent_tokens=total_tokens,
            token_effective_batch_size=token_eff,
            token_batch_utilization=utilization,
            total_token_work=total_work,
            max_token_work=max_work,
        )

    def _state_latent_tokens(self, state: DiffusionRequestState) -> int:
        explicit = _coerce_float(_extra_arg(state, "latent_tokens"))
        if explicit is None:
            explicit = _coerce_float(getattr(state.req.sampling_params, "latent_tokens", None))
        if explicit is not None and explicit > 0:
            return max(int(round(explicit)), 1)
        sampling = state.req.sampling_params
        width, height = _sampling_dimensions(sampling)
        frames = getattr(sampling, "num_frames", 1)
        if self.cost_model is not None:
            tokens = self.cost_model.latent_tokens(width, height, frames)
            if tokens is not None:
                return tokens
        if width is None or height is None:
            resolution = _coerce_float(getattr(sampling, "resolution", None)) or 1024.0
            width = width or resolution
            height = height or resolution
        try:
            frames_f = max(float(frames or 1), 1.0)
            return max(int(round((float(width) / self.token_patch_size) * (float(height) / self.token_patch_size) * frames_f)), 1)
        except (TypeError, ValueError):
            return 4096

    def _state_token_work(self, state: DiffusionRequestState) -> float:
        return self._state_latent_tokens(state) * _request_effective_size(state)

    def _states_from_snapshot_ids(self, sched_req_ids: Any) -> list[DiffusionRequestState]:
        if not isinstance(sched_req_ids, list):
            return []
        return [self._request_states[req_id] for req_id in sched_req_ids if req_id in self._request_states]

    def _states_matching_bucket(self, key: Any) -> list[DiffusionRequestState]:
        states = []
        for sched_req_id in [*self._running, *self._waiting]:
            state = self._request_states.get(sched_req_id)
            if state is None or state.is_finished():
                continue
            state_key = None if state.sampling_params_key is None else state.sampling_params_key.__dict__
            if state_key == key:
                states.append(state)
        return states[: self.max_num_running_reqs]


class AdaptiveTokenSloStepScheduler(TokenSloStepScheduler):
    """Token SLO scheduler with adaptive no-preemption admission guards."""

    def __init__(self) -> None:
        super().__init__()
        self.adaptive_min_laxity_ratio = 0.05
        self.adaptive_high_token_ratio = 0.75
        self.adaptive_priority_laxity_window_ms = 500.0
        self.adaptive_priority_ratio_window = 0.25
        self.adaptive_laxity_drop_margin_ms = 0.0

    def initialize(self, od_config) -> None:
        super().initialize(od_config)
        slo_config = _get_slo_scheduler_config(od_config)
        self.adaptive_min_laxity_ratio = _coerce_nonnegative_config(
            slo_config.get("adaptive_min_laxity_ratio"),
            0.05,
        )
        self.adaptive_high_token_ratio = max(
            _coerce_nonnegative_config(slo_config.get("adaptive_high_token_ratio"), 0.75),
            0.0,
        )
        self.adaptive_priority_laxity_window_ms = _coerce_nonnegative_config(
            slo_config.get("adaptive_priority_laxity_window_ms"),
            500.0,
        )
        self.adaptive_priority_ratio_window = _coerce_nonnegative_config(
            slo_config.get("adaptive_priority_ratio_window"),
            0.25,
        )
        self.adaptive_laxity_drop_margin_ms = _coerce_nonnegative_config(
            slo_config.get("adaptive_laxity_drop_margin_ms"),
            0.0,
        )

    def get_load_snapshot(self) -> dict[str, Any]:
        snapshot = super().get_load_snapshot()
        snapshot["policy"] = self.__class__.__name__
        now_s = float(snapshot.get("timestamp_s", 0.0) or 0.0)
        for bucket in snapshot.get("buckets", []):
            if not isinstance(bucket, dict):
                continue
            states = self._states_from_snapshot_ids(bucket.get("sched_req_ids"))
            if not states:
                continue
            step_ms = _coerce_float(bucket.get("estimated_step_ms"))
            if step_ms is None:
                step_ms = self._estimate_step_ms(states)
            bucket["min_laxity_ratio"] = self._bucket_min_laxity_ratio(states, step_ms, now_s)
        return snapshot

    def _selection_debug(self, candidates, selected, now_s: float) -> dict[str, Any]:
        debug = super()._selection_debug(candidates, selected, now_s)
        debug["policy"] = "adaptive_token_slo"
        for candidate in debug.get("candidates", []):
            states = self._states_from_snapshot_ids(candidate.get("sched_req_ids"))
            if not states:
                continue
            step_ms = self._estimate_step_ms(states)
            candidate["min_laxity_ratio"] = self._bucket_min_laxity_ratio(states, step_ms, now_s)
        return debug

    def _state_admission_priority(self, state: DiffusionRequestState, now_s: float):
        if self.adaptive_priority_laxity_window_ms <= 0:
            return super()._state_admission_priority(state, now_s)
        step_ms = self._estimate_step_ms([state])
        laxity_ms = self._state_laxity_ms(state, step_ms, now_s)
        has_deadline = 0.0 if math.isfinite(laxity_ms) else 1.0
        if math.isfinite(laxity_ms):
            laxity_rank = math.floor(laxity_ms / max(self.adaptive_priority_laxity_window_ms, 0.001))
        else:
            laxity_rank = math.inf
        ratio = self._state_laxity_ratio(state, step_ms, now_s)
        if math.isfinite(ratio) and self.adaptive_priority_ratio_window > 0:
            ratio_rank = math.floor(ratio / max(self.adaptive_priority_ratio_window, 0.001))
        else:
            ratio_rank = ratio
        return (
            has_deadline,
            laxity_rank,
            ratio_rank,
            -self._state_token_work(state),
            int(state.arrival_time_s * 1000),
        )

    def _would_violate_admission_guard(
        self,
        selected: list[DiffusionRequestState],
        state: DiffusionRequestState,
        now_s: float,
    ) -> bool:
        if super()._would_violate_admission_guard(selected, state, now_s):
            return True
        if not selected:
            return False

        candidate = [*selected, state]
        if not self._has_deadline(candidate):
            return False

        selected_step_ms = self._estimate_step_ms(selected)
        candidate_step_ms = self._estimate_step_ms(candidate)
        protected_states = self._adaptive_protected_states(selected, candidate)
        for protected in protected_states:
            before_laxity_ms = self._state_laxity_ms(protected, selected_step_ms, now_s)
            after_laxity_ms = self._state_laxity_ms(protected, candidate_step_ms, now_s)
            if (
                math.isfinite(before_laxity_ms)
                and before_laxity_ms >= self.min_laxity_guard_ms
                and after_laxity_ms < self.min_laxity_guard_ms
                and after_laxity_ms < before_laxity_ms - self.adaptive_laxity_drop_margin_ms
            ):
                return True

            before_ratio = self._state_laxity_ratio(protected, selected_step_ms, now_s)
            after_ratio = self._state_laxity_ratio(protected, candidate_step_ms, now_s)
            if (
                math.isfinite(before_ratio)
                and before_ratio >= self.adaptive_min_laxity_ratio
                and after_ratio < self.adaptive_min_laxity_ratio
                and after_ratio < before_ratio
            ):
                return True
        return False

    def _adaptive_protected_states(
        self,
        selected: list[DiffusionRequestState],
        candidate: list[DiffusionRequestState],
    ) -> list[DiffusionRequestState]:
        resident_ids = self._resident_sched_req_ids()
        max_tokens = max((self._state_latent_tokens(state) for state in candidate), default=1)
        protected: list[DiffusionRequestState] = []
        for state in selected:
            is_resident = state.sched_req_id in resident_ids
            is_high_token = self._state_latent_tokens(state) >= max_tokens * self.adaptive_high_token_ratio
            if is_resident or is_high_token:
                protected.append(state)
        return protected

    def _state_laxity_ratio(self, state: DiffusionRequestState, step_ms: float, now_s: float) -> float:
        if state.deadline_time_s is None:
            return math.inf
        remaining_work_ms = self._remaining_steps(state) * step_ms + self.decode_ms
        laxity_ms = (state.deadline_time_s - now_s) * 1000.0 - remaining_work_ms
        return laxity_ms / max(remaining_work_ms, 0.001)


class TokenStepPreemptiveSloStepScheduler(TokenSloStepScheduler):
    """Token SLO scheduler ablation that permits step-boundary preemption.

    This policy keeps PR4024 token-aware cost estimation, but it allows the
    scheduler to switch buckets between denoise steps when an urgent token-work
    bucket has meaningfully tighter completion-aware laxity. It does not
    interrupt a denoise step in flight.
    """

    def initialize(self, od_config) -> None:
        super().initialize(od_config)
        slo_config = _get_slo_scheduler_config(od_config)
        self.enable_step_preemption = _coerce_bool(slo_config.get("enable_step_preemption"), True)
        self.allow_preemptive_admission_swap = _coerce_bool(
            slo_config.get("allow_preemptive_admission_swap"),
            True,
        )

    def get_load_snapshot(self) -> dict[str, Any]:
        snapshot = super().get_load_snapshot()
        snapshot["policy"] = self.__class__.__name__
        snapshot["enable_step_preemption"] = self.enable_step_preemption
        snapshot["allow_preemptive_admission_swap"] = self.allow_preemptive_admission_swap
        return snapshot

    def _selection_debug(self, candidates, selected, now_s: float) -> dict[str, Any]:
        debug = super()._selection_debug(candidates, selected, now_s)
        debug["policy"] = "token_step_preemptive_slo"
        debug["enable_step_preemption"] = self.enable_step_preemption
        debug["allow_preemptive_admission_swap"] = self.allow_preemptive_admission_swap
        return debug

    def _can_admit_state(
        self,
        state: DiffusionRequestState,
        resident_ids: set[str],
        new_admissions: int,
    ) -> bool:
        if (
            self.enable_step_preemption
            and self.allow_preemptive_admission_swap
            and state.sched_req_id not in resident_ids
        ):
            return True
        return super()._can_admit_state(state, resident_ids, new_admissions)


def _coerce_nonnegative_config(value: Any, default: float) -> float:
    parsed = _coerce_float(value)
    if parsed is None or parsed < 0:
        return default
    return parsed


def _get_slo_scheduler_config(od_config: Any) -> dict[str, Any]:
    additional_config = getattr(od_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        return {}
    raw = additional_config.get("diffusion_slo_scheduler") or additional_config.get("slo_scheduler") or {}
    return raw if isinstance(raw, dict) else {}
