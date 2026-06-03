"""PARD State Planner for full-DAG latency estimates."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from vllm_omni.engine.dag_runtime import DagRuntimeConfig
from vllm_omni.engine.dag_types import (
    FULL_DAG_REQUIRED_STAGE_KINDS,
    DagRequestContext,
    DagStageKind,
    DagStageSpec,
)


class PardPlannerMode(StrEnum):
    NORMAL = "normal"
    LOWER = "lower"
    UPPER = "upper"
    BACK = "back"
    SUBSEQUENT_EXEC_ONLY = "subsequent_exec_only"


@dataclass(frozen=True, slots=True)
class PardStageLatencyProfile:
    """Per-stage latency lookup row used by PARD.

    queue_wait_ms and batch_wait_ms model Q_k and W_k. execution_ms models D_k.
    tail_quantile_ms represents the paper's downstream tail term
    F^{-1}_{k+1->N}(lambda) for the profiled stage/path. lower/upper fields are
    used for PARD-lower/PARD-upper ablations.
    """

    stage_kind: DagStageKind
    queue_wait_ms: float
    batch_wait_ms: float
    execution_ms: float
    tail_quantile_ms: float = 0.0
    tail_wait_quantiles_ms: dict[float, float] = field(default_factory=dict)
    throughput_work_per_s: float | None = None
    profile_version: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def execution_for_mode(self, _mode: PardPlannerMode) -> float:
        # PARD-lower/upper only bound the subsequent batch wait W_i. They must
        # not alter module execution durations d_i.
        return max(float(self.execution_ms), 0.0)

    def tail_wait_for_lambda(self, lambda_quantile: float) -> float:
        if not self.tail_wait_quantiles_ms:
            return max(float(self.tail_quantile_ms), 0.0)
        quantile = min(max(float(lambda_quantile), 0.0), 1.0)
        points = sorted((float(key), float(value)) for key, value in self.tail_wait_quantiles_ms.items())
        if quantile <= points[0][0]:
            return max(points[0][1], 0.0)
        if quantile >= points[-1][0]:
            return max(points[-1][1], 0.0)
        for (left_q, left_v), (right_q, right_v) in zip(points, points[1:], strict=False):
            if left_q <= quantile <= right_q:
                width = max(right_q - left_q, 0.001)
                ratio = (quantile - left_q) / width
                return max(left_v + ratio * (right_v - left_v), 0.0)
        return max(float(self.tail_quantile_ms), 0.0)


@dataclass(frozen=True, slots=True)
class PardStageLatencyEstimate:
    stage_id: int
    stage_kind: DagStageKind
    queue_wait_ms: float
    batch_wait_ms: float
    execution_ms: float
    tail_quantile_ms: float
    source: str
    profile_version: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def current_total_ms(self) -> float:
        return self.queue_wait_ms + self.batch_wait_ms + self.execution_ms

    @property
    def downstream_total_ms(self) -> float:
        return self.queue_wait_ms + self.batch_wait_ms + self.execution_ms + self.tail_quantile_ms


@dataclass(frozen=True, slots=True)
class PardLatencyPlan:
    request_id: str
    stage_id: int
    planner_mode: PardPlannerMode
    lambda_quantile: float
    l_pre_ms: float
    l_cur_ms: float
    l_sub_ms: float
    estimated_e2e_latency_ms: float
    remaining_budget_ms: float | None
    slo_budget_ms: float | None
    current_stage: PardStageLatencyEstimate
    downstream_path_ms_by_stage: dict[int, float]
    missing_profiles: tuple[DagStageKind, ...] = ()
    estimate_error_ms: float | None = None

    @property
    def would_miss_deadline(self) -> bool:
        if self.remaining_budget_ms is None:
            return False
        return self.l_cur_ms + self.l_sub_ms > self.remaining_budget_ms


class MissingPardProfilesError(RuntimeError):
    def __init__(self, missing: tuple[DagStageKind, ...]) -> None:
        self.missing = missing
        super().__init__("Missing PARD latency profiles: " + ", ".join(kind.value for kind in missing))


class MissingPardStagesError(RuntimeError):
    def __init__(self, missing: tuple[DagStageKind, ...]) -> None:
        self.missing = missing
        super().__init__("Missing full-DAG PARD stages: " + ", ".join(kind.value for kind in missing))


class PardStageLatencyLookup:
    """Profile/lookup holder for all PARD modules."""

    def __init__(
        self,
        profiles_by_kind: Mapping[DagStageKind, PardStageLatencyProfile] | None = None,
        *,
        profiles_by_stage_id: Mapping[int, PardStageLatencyProfile] | None = None,
    ) -> None:
        self._profiles_by_kind = dict(profiles_by_kind or {})
        self._profiles_by_stage_id = dict(profiles_by_stage_id or {})

    def profile_for(self, spec: DagStageSpec) -> PardStageLatencyProfile | None:
        return self._profiles_by_stage_id.get(spec.stage_id) or self._profiles_by_kind.get(spec.kind)

    def missing_required_profiles(
        self,
        stage_specs: tuple[DagStageSpec, ...],
        *,
        required_kinds: frozenset[DagStageKind] = FULL_DAG_REQUIRED_STAGE_KINDS,
    ) -> tuple[DagStageKind, ...]:
        present_kinds = {spec.kind for spec in stage_specs if spec.kind in required_kinds}
        missing: list[DagStageKind] = []
        for kind in sorted(present_kinds, key=lambda item: item.value):
            if kind not in self._profiles_by_kind and not any(
                spec.kind == kind and spec.stage_id in self._profiles_by_stage_id for spec in stage_specs
            ):
                missing.append(kind)
        return tuple(missing)

    def require_profiles(self, stage_specs: tuple[DagStageSpec, ...]) -> None:
        missing = self.missing_required_profiles(stage_specs)
        if missing:
            raise MissingPardProfilesError(missing)


def missing_required_stage_kinds(stage_specs: tuple[DagStageSpec, ...]) -> tuple[DagStageKind, ...]:
    present = {spec.kind for spec in stage_specs}
    missing = FULL_DAG_REQUIRED_STAGE_KINDS - present
    return tuple(sorted(missing, key=lambda item: item.value))


DitCostEstimator = Callable[[DagRequestContext, DagStageSpec, Mapping[str, Any] | None], tuple[float, str, dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class _DownstreamPathEstimate:
    queue_execution_ms: float
    execution_ms: float
    path_ms_by_stage: dict[int, float]


class PardStatePlanner:
    """Compute PARD L_pre + L_cur + L_sub for a request at stage entry."""

    def __init__(
        self,
        config: DagRuntimeConfig,
        latency_lookup: PardStageLatencyLookup,
        *,
        lambda_quantile: float = 0.1,
        mode: PardPlannerMode = PardPlannerMode.NORMAL,
        dit_cost_estimator: DitCostEstimator | None = None,
        require_full_profiles: bool = True,
    ) -> None:
        self.config = config
        self.latency_lookup = latency_lookup
        self.lambda_quantile = min(max(float(lambda_quantile), 0.0), 1.0)
        self.mode = mode
        self.dit_cost_estimator = dit_cost_estimator
        if require_full_profiles:
            missing_stages = missing_required_stage_kinds(config.stage_specs)
            if missing_stages:
                raise MissingPardStagesError(missing_stages)
            self.latency_lookup.require_profiles(config.stage_specs)

    def plan_stage_entry(
        self,
        ctx: DagRequestContext,
        stage_id: int,
        *,
        now_s: float | None = None,
        stage_snapshots: Mapping[int, Mapping[str, Any]] | None = None,
    ) -> PardLatencyPlan:
        now = time.time() if now_s is None else now_s
        spec = self.config.spec_by_id[stage_id]
        snapshots = stage_snapshots or {}
        missing = self.latency_lookup.missing_required_profiles(self.config.stage_specs)
        current = self.estimate_stage(ctx, spec, snapshots.get(stage_id))
        downstream = self._estimate_downstream_max_path(
            ctx,
            stage_id,
            snapshots=snapshots,
        )
        l_sub_ms = self._l_sub_for_mode(current, downstream)
        l_pre_ms = max((now - ctx.arrival_time_s) * 1000.0, 0.0)
        l_cur_ms = current.current_total_ms
        remaining_budget_ms = ctx.remaining_budget_ms(now_s=now)
        slo_budget_ms = None
        if ctx.deadline_time_s is not None:
            slo_budget_ms = max((ctx.deadline_time_s - ctx.arrival_time_s) * 1000.0, 0.0)
        observed_e2e_ms = _observed_elapsed_ms(ctx)
        estimated = l_pre_ms + l_cur_ms + l_sub_ms
        estimate_error_ms = None if observed_e2e_ms is None else observed_e2e_ms - estimated
        return PardLatencyPlan(
            request_id=ctx.request_id,
            stage_id=stage_id,
            planner_mode=self.mode,
            lambda_quantile=self.lambda_quantile,
            l_pre_ms=l_pre_ms,
            l_cur_ms=l_cur_ms,
            l_sub_ms=l_sub_ms,
            estimated_e2e_latency_ms=estimated,
            remaining_budget_ms=remaining_budget_ms,
            slo_budget_ms=slo_budget_ms,
            current_stage=current,
            downstream_path_ms_by_stage=dict(downstream.path_ms_by_stage),
            missing_profiles=missing,
            estimate_error_ms=estimate_error_ms,
        )

    def estimate_stage(
        self,
        ctx: DagRequestContext,
        spec: DagStageSpec,
        snapshot: Mapping[str, Any] | None = None,
    ) -> PardStageLatencyEstimate:
        profile = self.latency_lookup.profile_for(spec)
        if profile is None:
            raise MissingPardProfilesError((spec.kind,))
        queue_wait_ms = _snapshot_float(snapshot, "queue_wait_ms", profile.queue_wait_ms)
        batch_wait_ms = _snapshot_float(snapshot, "batch_wait_ms", profile.batch_wait_ms)
        execution_ms = profile.execution_for_mode(self.mode)
        source = f"profile:{profile.profile_version}"
        metadata: dict[str, Any] = dict(profile.metadata)
        if spec.kind == DagStageKind.DIT_DENOISE and self.dit_cost_estimator is not None:
            execution_ms, source, token_metadata = self.dit_cost_estimator(ctx, spec, snapshot)
            metadata.update(token_metadata)
        else:
            execution_ms = _snapshot_float(snapshot, "execution_ms", execution_ms)
        tail_quantile_ms = _snapshot_float(
            snapshot,
            "tail_quantile_ms",
            profile.tail_wait_for_lambda(self.lambda_quantile),
        )
        return PardStageLatencyEstimate(
            stage_id=spec.stage_id,
            stage_kind=spec.kind,
            queue_wait_ms=max(queue_wait_ms, 0.0),
            batch_wait_ms=max(batch_wait_ms, 0.0),
            execution_ms=max(execution_ms, 0.0),
            tail_quantile_ms=max(tail_quantile_ms, 0.0),
            source=source,
            profile_version=profile.profile_version,
            metadata=metadata,
        )

    def _l_sub_for_mode(
        self,
        current: PardStageLatencyEstimate,
        downstream: _DownstreamPathEstimate,
    ) -> float:
        if self.mode == PardPlannerMode.BACK:
            return 0.0
        if self.mode == PardPlannerMode.SUBSEQUENT_EXEC_ONLY:
            return downstream.execution_ms
        if self.mode == PardPlannerMode.LOWER:
            return downstream.queue_execution_ms
        if self.mode == PardPlannerMode.UPPER:
            return downstream.queue_execution_ms + downstream.execution_ms
        return downstream.queue_execution_ms + current.tail_quantile_ms

    def _estimate_downstream_max_path(
        self,
        ctx: DagRequestContext,
        stage_id: int,
        *,
        snapshots: Mapping[int, Mapping[str, Any]],
    ) -> _DownstreamPathEstimate:
        spec = self.config.spec_by_id[stage_id]
        if not spec.subs:
            return _DownstreamPathEstimate(queue_execution_ms=0.0, execution_ms=0.0, path_ms_by_stage={stage_id: 0.0})
        best = _DownstreamPathEstimate(queue_execution_ms=0.0, execution_ms=0.0, path_ms_by_stage={stage_id: 0.0})
        for sub_id in spec.subs:
            sub_spec = self.config.spec_by_id[sub_id]
            sub_estimate = self.estimate_stage(ctx, sub_spec, snapshots.get(sub_id))
            child = self._estimate_downstream_max_path(
                ctx,
                sub_id,
                snapshots=snapshots,
            )
            sub_queue_execution_ms = sub_estimate.queue_wait_ms + sub_estimate.execution_ms + child.queue_execution_ms
            sub_execution_ms = sub_estimate.execution_ms + child.execution_ms
            path_ms_by_stage = dict(child.path_ms_by_stage)
            path_ms_by_stage[sub_id] = sub_queue_execution_ms
            candidate = _DownstreamPathEstimate(
                queue_execution_ms=sub_queue_execution_ms,
                execution_ms=sub_execution_ms,
                path_ms_by_stage=path_ms_by_stage,
            )
            if candidate.queue_execution_ms > best.queue_execution_ms:
                best = candidate
        best.path_ms_by_stage[stage_id] = best.queue_execution_ms
        return best


def _snapshot_float(snapshot: Mapping[str, Any] | None, key: str, default: float) -> float:
    if snapshot is None:
        return float(default)
    value = snapshot.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _observed_elapsed_ms(ctx: DagRequestContext) -> float | None:
    completed = [
        lifecycle.completed_at_s
        for lifecycle in ctx.stage_lifecycle.values()
        if lifecycle.completed_at_s is not None
    ]
    if not completed:
        return None
    return max((max(completed) - ctx.arrival_time_s) * 1000.0, 0.0)


__all__ = [
    "MissingPardProfilesError",
    "MissingPardStagesError",
    "missing_required_stage_kinds",
    "PardLatencyPlan",
    "PardPlannerMode",
    "PardStageLatencyEstimate",
    "PardStageLatencyLookup",
    "PardStageLatencyProfile",
    "PardStatePlanner",
]
