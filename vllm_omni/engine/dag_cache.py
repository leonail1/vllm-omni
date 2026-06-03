"""Explicit cache/data-movement registry for the DAG runtime."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from vllm_omni.engine.dag_types import DagCacheMovementKind


@dataclass(frozen=True, slots=True)
class DagCacheLocation:
    stage_id: int
    replica_id: int | None = None
    address: str | None = None
    device: str | None = None


@dataclass(slots=True)
class DagCacheHandle:
    handle_id: str
    request_id: str
    producer_stage_id: int
    consumer_stage_ids: set[int]
    size_bytes: int | None = None
    location: DagCacheLocation | None = None
    created_at_s: float = field(default_factory=time.time)
    ready_at_s: float | None = None
    released: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DagCacheTraceEvent:
    handle_id: str
    request_id: str
    movement: DagCacheMovementKind
    producer_stage_id: int | None = None
    consumer_stage_id: int | None = None
    timestamp_s: float = field(default_factory=time.time)
    duration_ms: float | None = None
    prefetch_hit: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DagCacheRegistry:
    """Tracks cache handles, pull/prefetch activity, and cleanup."""

    def __init__(self) -> None:
        self._handles: dict[str, DagCacheHandle] = {}
        self._by_request: dict[str, set[str]] = {}
        self.trace: list[DagCacheTraceEvent] = []

    def create_handle(
        self,
        *,
        request_id: str,
        producer_stage_id: int,
        consumer_stage_ids: set[int] | frozenset[int],
        size_bytes: int | None = None,
        location: DagCacheLocation | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DagCacheHandle:
        handle_id = f"dag-cache-{uuid.uuid4().hex}"
        handle = DagCacheHandle(
            handle_id=handle_id,
            request_id=request_id,
            producer_stage_id=producer_stage_id,
            consumer_stage_ids=set(consumer_stage_ids),
            size_bytes=size_bytes,
            location=location,
            metadata=dict(metadata or {}),
        )
        self._handles[handle_id] = handle
        self._by_request.setdefault(request_id, set()).add(handle_id)
        self.trace.append(
            DagCacheTraceEvent(
                handle_id=handle_id,
                request_id=request_id,
                movement=DagCacheMovementKind.PUSH,
                producer_stage_id=producer_stage_id,
                metadata={"size_bytes": size_bytes, **(metadata or {})},
            )
        )
        return handle

    def mark_ready(self, handle_id: str, *, location: DagCacheLocation | None = None) -> DagCacheHandle:
        handle = self._handles[handle_id]
        handle.ready_at_s = time.time()
        if location is not None:
            handle.location = location
        return handle

    def prefetch(self, handle_id: str, *, consumer_stage_id: int, metadata: dict[str, Any] | None = None) -> None:
        handle = self._handles[handle_id]
        self.trace.append(
            DagCacheTraceEvent(
                handle_id=handle_id,
                request_id=handle.request_id,
                movement=DagCacheMovementKind.PREFETCH,
                producer_stage_id=handle.producer_stage_id,
                consumer_stage_id=consumer_stage_id,
                prefetch_hit=handle.ready_at_s is not None,
                metadata=dict(metadata or {}),
            )
        )

    def pull(
        self,
        handle_id: str,
        *,
        consumer_stage_id: int,
        start_s: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DagCacheHandle:
        handle = self._handles[handle_id]
        if consumer_stage_id not in handle.consumer_stage_ids:
            raise ValueError(f"stage {consumer_stage_id} is not a consumer for cache handle {handle_id}")
        now = time.time()
        start = now if start_s is None else start_s
        self.trace.append(
            DagCacheTraceEvent(
                handle_id=handle_id,
                request_id=handle.request_id,
                movement=DagCacheMovementKind.PULL,
                producer_stage_id=handle.producer_stage_id,
                consumer_stage_id=consumer_stage_id,
                duration_ms=max((now - start) * 1000.0, 0.0),
                prefetch_hit=handle.ready_at_s is not None and handle.ready_at_s <= start,
                metadata=dict(metadata or {}),
            )
        )
        return handle

    def release_request(self, request_id: str) -> list[str]:
        released: list[str] = []
        for handle_id in list(self._by_request.pop(request_id, set())):
            handle = self._handles.get(handle_id)
            if handle is None:
                continue
            handle.released = True
            released.append(handle_id)
        return released

    def get(self, handle_id: str) -> DagCacheHandle:
        return self._handles[handle_id]

    def snapshot(self) -> dict[str, Any]:
        return {
            "handles": {
                handle_id: {
                    "request_id": handle.request_id,
                    "producer_stage_id": handle.producer_stage_id,
                    "consumer_stage_ids": sorted(handle.consumer_stage_ids),
                    "size_bytes": handle.size_bytes,
                    "ready": handle.ready_at_s is not None,
                    "released": handle.released,
                }
                for handle_id, handle in self._handles.items()
            },
            "trace_events": len(self.trace),
        }
