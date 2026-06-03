"""PARD drop/cancel/cleanup execution controller.

Gate D maps broker drop decisions to concrete adapter operations. The
controller keeps trace state in the DAG request context so dropped requests do
not silently leak queues, cache handles, or bindings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from vllm_omni.engine.dag_runtime import DagRuntime
from vllm_omni.engine.dag_types import DagRequestContext, DagStageKind, DagStageStatus, DagTraceEvent
from vllm_omni.engine.pard_broker import PardBrokerDecision


class PardExecutionPlacement(StrEnum):
    QUEUED = "queued"
    BATCHED = "batched"
    RUNNING = "running"
    RUNNING_STEP = "running_step"
    COMPLETED = "completed"
    UNKNOWN = "unknown"


class PardDropCleanupOutcome(StrEnum):
    CLEANED_UP = "cleaned_up"
    PENDING_STEP_BOUNDARY = "pending_step_boundary"
    COMPLETED_NOOP = "completed_noop"
    ALREADY_CLEANED = "already_cleaned"


@dataclass(frozen=True, slots=True)
class PardDropCleanupResult:
    request_id: str
    stage_id: int
    placement: PardExecutionPlacement
    outcome: PardDropCleanupOutcome
    cancel_called_stage_ids: tuple[int, ...] = ()
    cleanup_called_stage_ids: tuple[int, ...] = ()
    released_cache_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pending_step_boundary(self) -> bool:
        return self.outcome == PardDropCleanupOutcome.PENDING_STEP_BOUNDARY


class PardDropController:
    """Executes PARD drop cleanup through DAG adapters."""

    def __init__(self, dag_runtime: DagRuntime) -> None:
        self.dag_runtime = dag_runtime

    async def apply_drop_decision(
        self,
        decision: PardBrokerDecision,
        *,
        now_s: float | None = None,
    ) -> PardDropCleanupResult:
        if not decision.dropped:
            raise ValueError("PARD cleanup requires a drop decision")
        ctx = self._get_required_context(decision.request_id)
        stage_id = decision.stage_id
        placement = self._placement(ctx, stage_id)
        now = time.time() if now_s is None else now_s
        if not ctx.dropped:
            ctx.mark_dropped(stage_id, decision.reason, now_s=now)
        if self._is_already_cleaned(ctx, stage_id):
            return self._record_result(
                ctx,
                stage_id,
                placement=placement,
                outcome=PardDropCleanupOutcome.ALREADY_CLEANED,
                now_s=now,
            )
        if placement == PardExecutionPlacement.COMPLETED:
            return self._record_result(
                ctx,
                stage_id,
                placement=placement,
                outcome=PardDropCleanupOutcome.COMPLETED_NOOP,
                now_s=now,
            )
        if self._requires_step_boundary(ctx, stage_id, placement):
            lifecycle = ctx.stage_lifecycle.get(stage_id)
            if lifecycle is not None:
                lifecycle.metadata["pard_drop_pending_step_boundary"] = True
                lifecycle.metadata["pard_drop_pending_reason"] = decision.reason
            return self._record_result(
                ctx,
                stage_id,
                placement=placement,
                outcome=PardDropCleanupOutcome.PENDING_STEP_BOUNDARY,
                now_s=now,
                metadata={"reason": decision.reason},
            )
        return await self._cancel_and_cleanup(ctx, stage_id, placement=placement, now_s=now)

    async def apply_pending_step_boundary(
        self,
        request_id: str,
        stage_id: int,
        *,
        now_s: float | None = None,
    ) -> PardDropCleanupResult:
        ctx = self._get_required_context(request_id)
        lifecycle = ctx.stage_lifecycle.get(stage_id)
        if lifecycle is None or not lifecycle.metadata.get("pard_drop_pending_step_boundary"):
            raise ValueError(f"request {request_id} has no pending step-boundary drop at stage {stage_id}")
        now = time.time() if now_s is None else now_s
        lifecycle.metadata["pard_drop_pending_step_boundary"] = False
        lifecycle.metadata["pard_step_boundary_drop_applied"] = True
        return await self._cancel_and_cleanup(
            ctx,
            stage_id,
            placement=PardExecutionPlacement.RUNNING_STEP,
            now_s=now,
            metadata={"step_boundary": True},
        )

    async def _cancel_and_cleanup(
        self,
        ctx: DagRequestContext,
        stage_id: int,
        *,
        placement: PardExecutionPlacement,
        now_s: float,
        metadata: dict[str, Any] | None = None,
    ) -> PardDropCleanupResult:
        cancel_called: list[int] = []
        adapter = self.dag_runtime.adapters.get(stage_id)
        if adapter is not None:
            await adapter.cancel(ctx.request_id)
            cancel_called.append(stage_id)
        cleanup_called: list[int] = []
        for adapter_stage_id, cleanup_adapter in sorted(self.dag_runtime.adapters.items()):
            await cleanup_adapter.cleanup(ctx.request_id)
            cleanup_called.append(adapter_stage_id)
        released_cache_refs = tuple(sorted(ctx.cache_refs))
        self.dag_runtime.cache_registry.release_request(ctx.request_id)
        lifecycle = ctx.stage_lifecycle.get(stage_id)
        if lifecycle is not None:
            lifecycle.status = DagStageStatus.DROPPED
            lifecycle.metadata["pard_cleanup_applied"] = True
        return self._record_result(
            ctx,
            stage_id,
            placement=placement,
            outcome=PardDropCleanupOutcome.CLEANED_UP,
            now_s=now_s,
            cancel_called_stage_ids=tuple(cancel_called),
            cleanup_called_stage_ids=tuple(cleanup_called),
            released_cache_refs=released_cache_refs,
            metadata=metadata,
        )

    def _get_required_context(self, request_id: str) -> DagRequestContext:
        ctx = self.dag_runtime.get_request_context(request_id)
        if ctx is None:
            raise KeyError(f"Unknown DAG request context: {request_id}")
        return ctx

    def _placement(self, ctx: DagRequestContext, stage_id: int) -> PardExecutionPlacement:
        lifecycle = ctx.stage_lifecycle.get(stage_id)
        if lifecycle is None:
            return PardExecutionPlacement.QUEUED
        explicit = lifecycle.metadata.get("pard_execution_placement")
        if explicit is not None:
            try:
                return PardExecutionPlacement(str(explicit))
            except ValueError:
                return PardExecutionPlacement.UNKNOWN
        if lifecycle.status == DagStageStatus.COMPLETED:
            return PardExecutionPlacement.COMPLETED
        if lifecycle.status == DagStageStatus.WAITING:
            return PardExecutionPlacement.QUEUED
        if lifecycle.status == DagStageStatus.RUNNING:
            if lifecycle.metadata.get("inside_denoise_step"):
                return PardExecutionPlacement.RUNNING_STEP
            if lifecycle.metadata.get("batch_admitted"):
                return PardExecutionPlacement.BATCHED
            return PardExecutionPlacement.RUNNING
        if lifecycle.status == DagStageStatus.DROPPED:
            previous_status = lifecycle.metadata.get("status_before_drop")
            if previous_status == DagStageStatus.COMPLETED.value:
                return PardExecutionPlacement.COMPLETED
            if previous_status == DagStageStatus.WAITING.value:
                return PardExecutionPlacement.QUEUED
            if previous_status == DagStageStatus.RUNNING.value:
                if lifecycle.metadata.get("inside_denoise_step"):
                    return PardExecutionPlacement.RUNNING_STEP
                if lifecycle.metadata.get("batch_admitted"):
                    return PardExecutionPlacement.BATCHED
                return PardExecutionPlacement.RUNNING
            return PardExecutionPlacement.UNKNOWN
        return PardExecutionPlacement.UNKNOWN

    def _requires_step_boundary(
        self,
        ctx: DagRequestContext,
        stage_id: int,
        placement: PardExecutionPlacement,
    ) -> bool:
        spec = self.dag_runtime.config.spec_by_id[stage_id]
        if spec.kind != DagStageKind.DIT_DENOISE:
            return False
        if not spec.batch.allow_step_boundary_preemption:
            return False
        if placement != PardExecutionPlacement.RUNNING_STEP:
            return False
        lifecycle = ctx.stage_lifecycle.get(stage_id)
        return not bool(lifecycle and lifecycle.metadata.get("at_step_boundary"))

    def _is_already_cleaned(self, ctx: DagRequestContext, stage_id: int) -> bool:
        lifecycle = ctx.stage_lifecycle.get(stage_id)
        return bool(lifecycle and lifecycle.metadata.get("pard_cleanup_applied"))

    def _record_result(
        self,
        ctx: DagRequestContext,
        stage_id: int,
        *,
        placement: PardExecutionPlacement,
        outcome: PardDropCleanupOutcome,
        now_s: float,
        cancel_called_stage_ids: tuple[int, ...] = (),
        cleanup_called_stage_ids: tuple[int, ...] = (),
        released_cache_refs: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> PardDropCleanupResult:
        result = PardDropCleanupResult(
            request_id=ctx.request_id,
            stage_id=stage_id,
            placement=placement,
            outcome=outcome,
            cancel_called_stage_ids=cancel_called_stage_ids,
            cleanup_called_stage_ids=cleanup_called_stage_ids,
            released_cache_refs=released_cache_refs,
            metadata=dict(metadata or {}),
        )
        ctx.trace.append(
            DagTraceEvent(
                request_id=ctx.request_id,
                stage_id=stage_id,
                event="pard_drop_cleanup",
                timestamp_s=now_s,
                details={
                    "placement": placement.value,
                    "outcome": outcome.value,
                    "cancel_called_stage_ids": list(cancel_called_stage_ids),
                    "cleanup_called_stage_ids": list(cleanup_called_stage_ids),
                    "released_cache_refs": list(released_cache_refs),
                    **(metadata or {}),
                },
            )
        )
        return result


__all__ = [
    "PardDropCleanupOutcome",
    "PardDropCleanupResult",
    "PardDropController",
    "PardExecutionPlacement",
]
