"""Full-DAG runtime foundation for vLLM-Omni orchestration."""

from __future__ import annotations

from collections import OrderedDict
import time
from dataclasses import dataclass
from typing import Any

from vllm_omni.engine.dag_cache import DagCacheLocation, DagCacheRegistry
from vllm_omni.engine.dag_types import (
    FULL_DAG_REQUIRED_STAGE_KINDS,
    DagRequestContext,
    DagStageBatchSpec,
    DagStageKind,
    DagStageResourceSpec,
    DagStageSpec,
)
from vllm_omni.engine.stage_adapters import StagePoolDagAdapter, build_stage_spec_from_pool


@dataclass(frozen=True, slots=True)
class DagRuntimeConfig:
    stage_specs: tuple[DagStageSpec, ...]
    entry_stage_ids: tuple[int, ...]
    exit_stage_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.stage_specs:
            raise ValueError("DAG runtime requires at least one stage")
        ids = [spec.stage_id for spec in self.stage_specs]
        if len(ids) != len(set(ids)):
            raise ValueError("DAG stage ids must be unique")
        spec_by_id = {spec.stage_id: spec for spec in self.stage_specs}
        for spec in self.stage_specs:
            for pre in spec.pres:
                if pre not in spec_by_id:
                    raise ValueError(f"stage {spec.stage_id} has missing predecessor {pre}")
                if spec.stage_id not in spec_by_id[pre].subs:
                    raise ValueError(f"stage {spec.stage_id} predecessor {pre} missing reverse edge")
            for sub in spec.subs:
                if sub not in spec_by_id:
                    raise ValueError(f"stage {spec.stage_id} has missing successor {sub}")
                if spec.stage_id not in spec_by_id[sub].pres:
                    raise ValueError(f"stage {spec.stage_id} successor {sub} missing reverse edge")
        if set(self.entry_stage_ids) != {spec.stage_id for spec in self.stage_specs if not spec.pres}:
            raise ValueError("entry_stage_ids must match stages without predecessors")
        if set(self.exit_stage_ids) != {spec.stage_id for spec in self.stage_specs if not spec.subs}:
            raise ValueError("exit_stage_ids must match stages without successors")
        _topological_order(self.stage_specs)

    @property
    def spec_by_id(self) -> dict[int, DagStageSpec]:
        return {spec.stage_id: spec for spec in self.stage_specs}


def _topological_order(stage_specs: tuple[DagStageSpec, ...]) -> tuple[int, ...]:
    by_id = {spec.stage_id: spec for spec in stage_specs}
    incoming = {spec.stage_id: set(spec.pres) for spec in stage_specs}
    ready = sorted(stage_id for stage_id, pres in incoming.items() if not pres)
    order: list[int] = []
    while ready:
        stage_id = ready.pop(0)
        order.append(stage_id)
        for sub in by_id[stage_id].subs:
            incoming[sub].discard(stage_id)
            if not incoming[sub] and sub not in order and sub not in ready:
                ready.append(sub)
                ready.sort()
    if len(order) != len(stage_specs):
        raise ValueError("DAG stage graph must be acyclic")
    return tuple(order)


def _stage_pool_predecessors(stage_pools: list[Any]) -> dict[int, tuple[int, ...]]:
    stage_ids = {int(pool.stage_id) for pool in stage_pools}
    predecessors: dict[int, tuple[int, ...]] = {}
    any_declared_edges = False
    has_any_source_metadata = any(
        getattr(getattr(pool, "stage_client", None), "engine_input_source", None)
        for pool in stage_pools
    )
    for index, pool in enumerate(stage_pools):
        stage_id = int(pool.stage_id)
        client = getattr(pool, "stage_client", None)
        raw_sources = getattr(client, "engine_input_source", None)
        if raw_sources:
            normalized_sources: set[int] = set()
            for source in raw_sources:
                try:
                    source_id = int(source)
                except (TypeError, ValueError):
                    continue
                if source_id == stage_id:
                    continue
                if source_id in stage_ids:
                    normalized_sources.add(source_id)
            sources = tuple(sorted(normalized_sources))
            any_declared_edges = any_declared_edges or bool(sources)
        elif index > 0 and not has_any_source_metadata:
            sources = (int(stage_pools[index - 1].stage_id),)
        else:
            sources = ()
        predecessors[stage_id] = sources
    if any_declared_edges:
        return predecessors
    return predecessors


def build_dag_config_from_stage_pools(stage_pools: list[Any]) -> DagRuntimeConfig:
    predecessors = _stage_pool_predecessors(stage_pools)
    successors: dict[int, list[int]] = {int(pool.stage_id): [] for pool in stage_pools}
    for stage_id, pres in predecessors.items():
        for pre in pres:
            successors.setdefault(pre, []).append(stage_id)

    specs: list[DagStageSpec] = []
    for pool in stage_pools:
        stage_id = int(pool.stage_id)
        specs.append(
            build_stage_spec_from_pool(
                pool,
                pres=predecessors[stage_id],
                subs=tuple(sorted(successors.get(stage_id, []))),
            )
        )
    return DagRuntimeConfig(
        stage_specs=tuple(specs),
        entry_stage_ids=tuple(spec.stage_id for spec in specs if not spec.pres),
        exit_stage_ids=tuple(spec.stage_id for spec in specs if not spec.subs),
    )


def build_linear_dag_config_from_stage_pools(stage_pools: list[Any]) -> DagRuntimeConfig:
    return build_dag_config_from_stage_pools(stage_pools)


def build_required_full_dag_template(resource_by_kind: dict[DagStageKind, DagStageResourceSpec]) -> DagRuntimeConfig:
    """Build the required six-role DAG template for tests and config validation."""

    default_resource = DagStageResourceSpec(replica_count=1, cards_per_replica=1)

    def resource(kind: DagStageKind) -> DagStageResourceSpec:
        return resource_by_kind.get(kind, default_resource)

    specs = (
        DagStageSpec(0, "text_encoder", DagStageKind.TEXT_ENCODER, (), (3,), resource(DagStageKind.TEXT_ENCODER)),
        DagStageSpec(1, "image_encoder", DagStageKind.IMAGE_ENCODER, (), (2,), resource(DagStageKind.IMAGE_ENCODER)),
        DagStageSpec(2, "vae_encoder", DagStageKind.VAE_ENCODER, (1,), (3,), resource(DagStageKind.VAE_ENCODER)),
        DagStageSpec(
            3,
            "dit_denoise",
            DagStageKind.DIT_DENOISE,
            (0, 2),
            (4, 5),
            resource(DagStageKind.DIT_DENOISE),
            batch=DagStageBatchSpec(allow_step_boundary_preemption=True),
        ),
        DagStageSpec(4, "vae_decoder", DagStageKind.VAE_DECODER, (3,), (), resource(DagStageKind.VAE_DECODER)),
        DagStageSpec(5, "audio_decoder", DagStageKind.AUDIO_DECODER, (3,), (), resource(DagStageKind.AUDIO_DECODER)),
    )
    return DagRuntimeConfig(stage_specs=specs, entry_stage_ids=(0, 1), exit_stage_ids=(4, 5))


class DagRuntime:
    """Request lifecycle and trace holder for the DAG control plane."""

    def __init__(
        self,
        config: DagRuntimeConfig,
        adapters: dict[int, Any] | None = None,
        *,
        completed_trace_limit: int = 256,
    ) -> None:
        self.config = config
        self.adapters: dict[int, Any] = dict(adapters or {})
        self.cache_registry = DagCacheRegistry()
        self.request_contexts: dict[str, DagRequestContext] = {}
        self.completed_trace_limit = max(int(completed_trace_limit), 0)
        self.completed_request_traces: OrderedDict[str, list[Any]] = OrderedDict()

    @classmethod
    def from_stage_pools(
        cls,
        stage_pools: list[Any],
        *,
        completed_trace_limit: int = 256,
    ) -> "DagRuntime":
        config = build_dag_config_from_stage_pools(stage_pools)
        adapters = {
            spec.stage_id: StagePoolDagAdapter(pool, spec)
            for spec, pool in zip(config.stage_specs, stage_pools, strict=True)
        }
        return cls(config, adapters, completed_trace_limit=completed_trace_limit)

    def validate_required_full_dag_roles(self) -> set[DagStageKind]:
        present = {spec.kind for spec in self.config.stage_specs}
        return set(FULL_DAG_REQUIRED_STAGE_KINDS - present)

    def create_request_context(
        self,
        request_id: str,
        *,
        arrival_time_s: float | None = None,
        deadline_time_s: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DagRequestContext:
        ctx = DagRequestContext(
            request_id=request_id,
            arrival_time_s=time.time() if arrival_time_s is None else arrival_time_s,
            deadline_time_s=deadline_time_s,
            metadata=dict(metadata or {}),
        )
        self.request_contexts[request_id] = ctx
        return ctx

    def get_request_context(self, request_id: str) -> DagRequestContext | None:
        return self.request_contexts.get(request_id)

    def enter_stage(self, request_id: str, stage_id: int, *, replica_id: int | None = None) -> None:
        ctx = self.request_contexts.get(request_id)
        if ctx is not None:
            ctx.enter_stage(stage_id, replica_id=replica_id)

    def finish_stage(
        self,
        request_id: str,
        stage_id: int,
        *,
        execution_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        ctx = self.request_contexts.get(request_id)
        if ctx is not None:
            ctx.finish_stage(stage_id, execution_ms=execution_ms, metadata=metadata)

    def create_cache_ref(
        self,
        request_id: str,
        *,
        producer_stage_id: int,
        consumer_stage_ids: set[int] | frozenset[int],
        size_bytes: int | None = None,
        location: DagCacheLocation | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        handle = self.cache_registry.create_handle(
            request_id=request_id,
            producer_stage_id=producer_stage_id,
            consumer_stage_ids=consumer_stage_ids,
            size_bytes=size_bytes,
            location=location,
            metadata=metadata,
        )
        ctx = self.request_contexts.get(request_id)
        if ctx is not None:
            ctx.add_cache_ref(handle.handle_id, stage_id=producer_stage_id, metadata=metadata)
        return handle.handle_id

    def mark_dropped(self, request_id: str, *, stage_id: int, reason: str) -> None:
        ctx = self.request_contexts.get(request_id)
        if ctx is not None:
            ctx.mark_dropped(stage_id, reason)

    def cleanup_request(self, request_id: str) -> None:
        self.cache_registry.release_request(request_id)
        self.request_contexts.pop(request_id, None)

    def archive_request_trace(self, request_id: str) -> None:
        if self.completed_trace_limit <= 0:
            return
        ctx = self.request_contexts.get(request_id)
        if ctx is None:
            return
        self.completed_request_traces[request_id] = list(ctx.trace)
        self.completed_request_traces.move_to_end(request_id)
        while len(self.completed_request_traces) > self.completed_trace_limit:
            self.completed_request_traces.popitem(last=False)

    def get_completed_trace(self, request_id: str) -> list[Any] | None:
        trace = self.completed_request_traces.get(request_id)
        return None if trace is None else list(trace)

    def snapshot(self) -> dict[str, Any]:
        return {
            "stages": [
                {
                    "stage_id": spec.stage_id,
                    "kind": spec.kind.value,
                    "pres": list(spec.pres),
                    "subs": list(spec.subs),
                    "replicas": spec.resource.replica_count,
                    "cards_per_replica": spec.resource.cards_per_replica,
                }
                for spec in self.config.stage_specs
            ],
            "requests": len(self.request_contexts),
            "completed_request_traces": len(self.completed_request_traces),
            "completed_trace_limit": self.completed_trace_limit,
            "cache": self.cache_registry.snapshot(),
        }
