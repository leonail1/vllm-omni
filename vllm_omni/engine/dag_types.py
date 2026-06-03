"""Core data types for the vLLM-Omni DAG runtime.

These types are deliberately scheduler-agnostic. Gate B only models the
end-to-end DAG, request lifecycle, resource layout, and stage IO. PARD drop
decisions are introduced later after the full DAG state is available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DagStageKind(StrEnum):
    """Logical work units required by the full-DAG scheduler plan."""

    TEXT_ENCODER = "text_encoder"
    IMAGE_ENCODER = "image_encoder"
    VAE_ENCODER = "vae_encoder"
    DIT_DENOISE = "dit_denoise"
    VAE_DECODER = "vae_decoder"
    AUDIO_DECODER = "audio_decoder"
    GENERIC_LLM = "generic_llm"
    GENERIC_DIFFUSION = "generic_diffusion"


FULL_DAG_REQUIRED_STAGE_KINDS: frozenset[DagStageKind] = frozenset(
    {
        DagStageKind.TEXT_ENCODER,
        DagStageKind.IMAGE_ENCODER,
        DagStageKind.VAE_ENCODER,
        DagStageKind.DIT_DENOISE,
        DagStageKind.VAE_DECODER,
        DagStageKind.AUDIO_DECODER,
    }
)


class DagStageStatus(StrEnum):
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    DROPPED = "dropped"


class DagCacheMovementKind(StrEnum):
    PUSH = "push"
    PULL = "pull"
    PREFETCH = "prefetch"


@dataclass(frozen=True, slots=True)
class DagStageReplicaSpec:
    replica_id: int
    device_ids: tuple[str, ...] = ()

    @property
    def card_count(self) -> int:
        return len(self.device_ids)


@dataclass(frozen=True, slots=True)
class DagStageResourceSpec:
    replica_count: int
    cards_per_replica: int
    replicas: tuple[DagStageReplicaSpec, ...] = ()
    queue_capacity: int | None = None
    max_batch_size: int | None = None

    def __post_init__(self) -> None:
        if self.replica_count <= 0:
            raise ValueError("replica_count must be positive")
        if self.cards_per_replica <= 0:
            raise ValueError("cards_per_replica must be positive")
        if self.replicas and len(self.replicas) != self.replica_count:
            raise ValueError("replicas length must match replica_count")


@dataclass(frozen=True, slots=True)
class DagStageBatchSpec:
    admission_policy: str = "stage_default"
    scheduler_policy: str | None = None
    allow_step_boundary_preemption: bool = False
    max_batch_size: int | None = None


@dataclass(frozen=True, slots=True)
class DagStageSpec:
    stage_id: int
    name: str
    kind: DagStageKind
    pres: tuple[int, ...]
    subs: tuple[int, ...]
    resource: DagStageResourceSpec
    batch: DagStageBatchSpec = field(default_factory=DagStageBatchSpec)
    adapter_name: str = "stage_pool"
    stage_type: str | None = None
    final_output: bool = False
    final_output_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage_id < 0:
            raise ValueError("stage_id must be non-negative")
        if not self.name:
            raise ValueError("stage name must be non-empty")
        if self.stage_id in self.pres or self.stage_id in self.subs:
            raise ValueError("stage cannot depend on itself")


@dataclass(frozen=True, slots=True)
class DagStageInput:
    request_id: str
    stage_id: int
    payload: Any
    cache_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DagStageOutput:
    request_id: str
    stage_id: int
    payload: Any
    cache_refs: tuple[str, ...] = ()
    finished: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DagStageLifecycle:
    stage_id: int
    status: DagStageStatus = DagStageStatus.WAITING
    first_entered_at_s: float | None = None
    last_entered_at_s: float | None = None
    completed_at_s: float | None = None
    replica_id: int | None = None
    wait_ms: float | None = None
    execution_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DagTraceEvent:
    request_id: str
    event: str
    stage_id: int | None = None
    timestamp_s: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DagRequestContext:
    request_id: str
    arrival_time_s: float
    deadline_time_s: float | None = None
    current_stage_id: int | None = None
    completed_stage_ids: set[int] = field(default_factory=set)
    stage_lifecycle: dict[int, DagStageLifecycle] = field(default_factory=dict)
    cache_refs: set[str] = field(default_factory=set)
    dropped: bool = False
    drop_stage_id: int | None = None
    drop_reason: str | None = None
    trace: list[DagTraceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def remaining_budget_ms(self, now_s: float | None = None) -> float | None:
        if self.deadline_time_s is None:
            return None
        now = time.time() if now_s is None else now_s
        return (self.deadline_time_s - now) * 1000.0

    def enter_stage(self, stage_id: int, *, replica_id: int | None = None, now_s: float | None = None) -> None:
        now = time.time() if now_s is None else now_s
        lifecycle = self.stage_lifecycle.setdefault(stage_id, DagStageLifecycle(stage_id=stage_id))
        if lifecycle.first_entered_at_s is None:
            lifecycle.first_entered_at_s = now
        lifecycle.last_entered_at_s = now
        lifecycle.status = DagStageStatus.RUNNING
        lifecycle.replica_id = replica_id
        self.current_stage_id = stage_id
        self.trace.append(
            DagTraceEvent(
                request_id=self.request_id,
                stage_id=stage_id,
                event="stage_enter",
                timestamp_s=now,
                details={"replica_id": replica_id},
            )
        )

    def finish_stage(
        self,
        stage_id: int,
        *,
        now_s: float | None = None,
        execution_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = time.time() if now_s is None else now_s
        lifecycle = self.stage_lifecycle.setdefault(stage_id, DagStageLifecycle(stage_id=stage_id))
        lifecycle.status = DagStageStatus.COMPLETED
        lifecycle.completed_at_s = now
        lifecycle.execution_ms = execution_ms
        if metadata:
            lifecycle.metadata.update(metadata)
        self.completed_stage_ids.add(stage_id)
        if self.current_stage_id == stage_id:
            self.current_stage_id = None
        self.trace.append(
            DagTraceEvent(
                request_id=self.request_id,
                stage_id=stage_id,
                event="stage_finish",
                timestamp_s=now,
                details={"execution_ms": execution_ms, **(metadata or {})},
            )
        )

    def add_cache_ref(self, cache_ref: str, *, stage_id: int | None = None, metadata: dict[str, Any] | None = None) -> None:
        self.cache_refs.add(cache_ref)
        self.trace.append(
            DagTraceEvent(
                request_id=self.request_id,
                stage_id=stage_id,
                event="cache_ref",
                details={"cache_ref": cache_ref, **(metadata or {})},
            )
        )

    def mark_dropped(self, stage_id: int, reason: str, *, now_s: float | None = None) -> None:
        now = time.time() if now_s is None else now_s
        self.dropped = True
        self.drop_stage_id = stage_id
        self.drop_reason = reason
        lifecycle = self.stage_lifecycle.setdefault(stage_id, DagStageLifecycle(stage_id=stage_id))
        lifecycle.status = DagStageStatus.DROPPED
        self.trace.append(
            DagTraceEvent(
                request_id=self.request_id,
                stage_id=stage_id,
                event="request_drop",
                timestamp_s=now,
                details={"reason": reason},
            )
        )
