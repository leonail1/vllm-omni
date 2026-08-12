# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage-2 data-plane backends for chunked diffusion weight transport."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from vllm_omni.platforms import current_omni_platform

from .chunked_transport import (
    ChunkMeta,
    PartManifest,
    SourceLayout,
    TransportBackendKind,
    metrics_enabled,
)


@dataclass(frozen=True)
class TransportCapability:
    world_size: int
    rank: int
    global_ranks: tuple[int, ...]
    same_host: bool
    p2p_supported: bool
    p2p_matrix: tuple[tuple[bool, ...], ...] = ()
    pair_ranks: tuple[tuple[int, int], ...] = ()
    pipeline_ranks: tuple[int, ...] = ()
    pair_group: Any | None = None
    pipeline_hop_groups: tuple[Any | None, ...] = ()
    native_persistent: bool = False
    topology: str = "unknown"


@dataclass(frozen=True)
class SupportResult:
    supported: bool
    reason: str | None = None


@dataclass(frozen=True)
class TransportSelection:
    requested_backend: TransportBackendKind
    effective_backend: TransportBackendKind
    requested_source_layout: SourceLayout
    effective_source_layout: SourceLayout
    fallback_reason: str | None = None


@dataclass(frozen=True)
class TransportStreams:
    copy: Any
    communication: Any


@dataclass(frozen=True)
class ChunkEvents:
    h2d_done: Any
    transport_done: Any
    input_reusable: Any | None = None
    relay_done: Any | None = None


@dataclass(frozen=True)
class ChunkCompletion:
    event: Any
    stream: Any
    works: tuple[Any, ...] = ()


@dataclass
class BackendCounters:
    submitted_parts: int = 0
    submitted_chunks: int = 0
    host_h2d_bytes: int = 0
    fabric_bytes: int = 0
    p2p_hops: int = 0
    schedule_builds: int = 0
    schedule_replays: int = 0
    async_works: int = 0
    backend_chunks: dict[str, int] = field(default_factory=dict)


TraceFactory = Callable[[str], AbstractContextManager[Any]]


def _pair_ranks(capability: TransportCapability) -> tuple[tuple[int, int], ...]:
    if capability.pair_ranks:
        return capability.pair_ranks
    return tuple((rank, rank + 1) for rank in range(0, capability.world_size, 2))


def _pipeline_ranks(capability: TransportCapability) -> tuple[int, ...]:
    return capability.pipeline_ranks or tuple(range(capability.world_size))


def _edge_supported(capability: TransportCapability, src: int, dst: int) -> bool:
    if not capability.p2p_matrix:
        return capability.p2p_supported
    return capability.p2p_matrix[src][dst] and capability.p2p_matrix[dst][src]


class WeightTransportBackend(Protocol):
    kind: TransportBackendKind
    source_layout: SourceLayout
    requires_local_input: bool
    counters: BackendCounters

    def supports(self, capability: TransportCapability, plan: PartManifest | None = None) -> SupportResult: ...

    def begin_part(self, streams: TransportStreams, prior_last_use: Any | None) -> None: ...

    def submit_chunk(
        self,
        *,
        source: torch.Tensor | None,
        local_input: torch.Tensor | None,
        full_output: torch.Tensor,
        chunk_meta: ChunkMeta,
        streams: TransportStreams,
        events: ChunkEvents,
        group: torch.distributed.ProcessGroup | None,
        generation: int,
        non_blocking: bool,
        trace: TraceFactory,
    ) -> ChunkCompletion: ...

    def finalize_part(
        self,
        completions: Sequence[ChunkCompletion],
        *,
        ready_event: Any,
        streams: TransportStreams,
    ) -> Any: ...

    def reset_generation(self, generation: int) -> None: ...

    def reset_counters(self) -> None: ...

    def close(self) -> None: ...


def _support_backend(
    backend: TransportBackendKind,
    source_layout: SourceLayout,
    capability: TransportCapability,
) -> SupportResult:
    if backend is TransportBackendKind.REFERENCE:
        if source_layout is SourceLayout.FS_SHARDED_HOST:
            return SupportResult(True)
        return SupportResult(False, "reference requires fs_sharded_host")

    if backend is TransportBackendKind.GROUP_SCATTER_AG:
        if source_layout is not SourceLayout.FS_SHARDED_HOST:
            return SupportResult(False, "group_scatter_ag requires fs_sharded_host")
        if capability.world_size <= 1:
            return SupportResult(False, "group_scatter_ag requires an FS group larger than one rank")
        return SupportResult(True)

    if backend is TransportBackendKind.GROUP_PERSISTENT:
        if source_layout is not SourceLayout.FS_SHARDED_HOST:
            return SupportResult(False, "group_persistent requires fs_sharded_host")
        if capability.world_size <= 1:
            return SupportResult(False, "group_persistent requires an FS group larger than one rank")
        if not capability.native_persistent:
            return SupportResult(False, "group_persistent requires validated native graph/kernel support")
        return SupportResult(True)

    if backend is TransportBackendKind.PAIR_COPY:
        if source_layout is not SourceLayout.PAIR_LEADER_FULL_HOST:
            return SupportResult(False, "pair_copy requires pair_leader_full_host")
        if capability.world_size <= 1 or capability.world_size % 2:
            return SupportResult(False, "pair_copy requires an even FS group larger than one rank")
        if not capability.same_host or not capability.p2p_supported:
            return SupportResult(False, "pair_copy requires validated same-host P2P edges")
        if any(not _edge_supported(capability, leader, peer) for leader, peer in _pair_ranks(capability)):
            return SupportResult(False, "pair_copy requires every selected pair edge to support P2P")
        return SupportResult(True)

    if backend is TransportBackendKind.GROUP_PIPELINE_MEMCPY:
        if source_layout is not SourceLayout.GROUP_OWNER_FULL_HOST:
            return SupportResult(False, "group_pipeline_memcpy requires group_owner_full_host")
        if capability.world_size <= 1:
            return SupportResult(False, "group_pipeline_memcpy requires an FS group larger than one rank")
        if not capability.same_host or not capability.p2p_supported:
            return SupportResult(False, "group_pipeline_memcpy requires a validated same-host P2P chain")
        order = _pipeline_ranks(capability)
        if len(order) != capability.world_size or set(order) != set(range(capability.world_size)):
            return SupportResult(False, "pipeline rank order must contain every local FS rank exactly once")
        if order[0] != 0:
            return SupportResult(False, "pipeline rank order must start at owner local rank 0")
        if any(not _edge_supported(capability, src, dst) for src, dst in zip(order, order[1:])):
            return SupportResult(False, "group_pipeline_memcpy requires every planned hop to support P2P")
        return SupportResult(True)

    return SupportResult(False, f"unsupported backend: {backend.value}")


def select_transport(
    requested_backend: TransportBackendKind,
    requested_source_layout: SourceLayout,
    capability: TransportCapability,
) -> TransportSelection:
    if requested_backend is TransportBackendKind.AUTO:
        # group_persistent stays opt-in: it is only justified after a
        # reference trace proves launch overhead (design section 15.2), so
        # auto selects the safe chunked AllGather schedule.
        candidate = {
            SourceLayout.FS_SHARDED_HOST: TransportBackendKind.GROUP_SCATTER_AG,
            SourceLayout.PAIR_LEADER_FULL_HOST: TransportBackendKind.PAIR_COPY,
            SourceLayout.GROUP_OWNER_FULL_HOST: TransportBackendKind.GROUP_PIPELINE_MEMCPY,
        }[requested_source_layout]
        if capability.world_size <= 1:
            candidate = TransportBackendKind.REFERENCE
    else:
        candidate = requested_backend

    support = _support_backend(candidate, requested_source_layout, capability)
    if support.supported:
        return TransportSelection(
            requested_backend=requested_backend,
            effective_backend=candidate,
            requested_source_layout=requested_source_layout,
            effective_source_layout=requested_source_layout,
        )

    fallback_backend = (
        TransportBackendKind.GROUP_SCATTER_AG if capability.world_size > 1 else TransportBackendKind.REFERENCE
    )
    fallback_source = SourceLayout.FS_SHARDED_HOST

    return TransportSelection(
        requested_backend=requested_backend,
        effective_backend=fallback_backend,
        requested_source_layout=requested_source_layout,
        effective_source_layout=fallback_source,
        fallback_reason=(
            f"{candidate.value} is unsupported: {support.reason}; "
            f"using {fallback_backend.value} with {fallback_source.value}"
        ),
    )


class _BaseBackend:
    kind = TransportBackendKind.REFERENCE
    source_layout = SourceLayout.FS_SHARDED_HOST
    requires_local_input = True
    writes_output_on_copy = False

    def __init__(self, capability: TransportCapability) -> None:
        self.capability = capability
        self.counters = BackendCounters()
        # Cached at construction: counter updates cost one predictable
        # branch per chunk when the metrics gate is off.
        self._metrics_on = metrics_enabled()
        self._generation = -1
        self._closed = False

    def supports(self, capability: TransportCapability, plan: PartManifest | None = None) -> SupportResult:
        del plan
        return _support_backend(self.kind, self.source_layout, capability)

    def begin_part(self, streams: TransportStreams, prior_last_use: Any | None) -> None:
        if self._closed:
            raise RuntimeError("weight transport backend is closed")
        if self._metrics_on:
            self.counters.submitted_parts += 1
        if prior_last_use is None:
            return
        streams.communication.wait_event(prior_last_use)
        if self.writes_output_on_copy:
            streams.copy.wait_event(prior_last_use)

    def _count_chunk(
        self,
        host_bytes: int,
        fabric_bytes: int,
        *,
        p2p_hops: int = 0,
        async_works: int = 0,
    ) -> None:
        if not self._metrics_on:
            return
        self.counters.submitted_chunks += 1
        self.counters.host_h2d_bytes += host_bytes
        self.counters.fabric_bytes += fabric_bytes
        self.counters.p2p_hops += p2p_hops
        self.counters.async_works += async_works
        key = self.kind.value
        self.counters.backend_chunks[key] = self.counters.backend_chunks.get(key, 0) + 1

    def finalize_part(
        self,
        completions: Sequence[ChunkCompletion],
        *,
        ready_event: Any,
        streams: TransportStreams,
    ) -> Any:
        for completion in completions:
            for work in completion.works:
                work.wait()
        stream = completions[-1].stream if completions else streams.communication
        with current_omni_platform.stream(stream):
            ready_event.record(stream)
        return ready_event

    def reset_generation(self, generation: int) -> None:
        if generation < self._generation:
            raise RuntimeError(f"transport generation moved backwards: {generation} < {self._generation}")
        self._generation = generation

    def reset_counters(self) -> None:
        self.counters = BackendCounters()

    def close(self) -> None:
        self._closed = True


class ReferenceBackend(_BaseBackend):
    kind = TransportBackendKind.REFERENCE
    _transport_trace_name = "all_gather"

    def __init__(self, capability: TransportCapability) -> None:
        super().__init__(capability)
        self.requires_local_input = capability.world_size > 1
        self.writes_output_on_copy = capability.world_size <= 1

    def submit_chunk(
        self,
        *,
        source: torch.Tensor | None,
        local_input: torch.Tensor | None,
        full_output: torch.Tensor,
        chunk_meta: ChunkMeta,
        streams: TransportStreams,
        events: ChunkEvents,
        group: torch.distributed.ProcessGroup | None,
        generation: int,
        non_blocking: bool,
        trace: TraceFactory,
    ) -> ChunkCompletion:
        del generation
        if source is None:
            raise RuntimeError("reference transport requires a local Host source")

        if self.capability.world_size <= 1:
            if source.numel() != chunk_meta.padded_numel:
                raise RuntimeError("single-rank reference transport requires one full padded Host chunk")
            with current_omni_platform.stream(streams.copy):
                with trace("h2d"):
                    full_output.copy_(source, non_blocking=non_blocking)
                events.h2d_done.record(streams.copy)
            self._count_chunk(source.numel() * source.element_size(), 0)
            return ChunkCompletion(events.h2d_done, streams.copy)

        if local_input is None or group is None:
            raise RuntimeError("reference FS transport requires a local input buffer and process group")
        return self._submit_fs_chunk(
            source=source,
            local_input=local_input,
            full_output=full_output,
            chunk_meta=chunk_meta,
            streams=streams,
            events=events,
            non_blocking=non_blocking,
            trace=trace,
            collective=lambda: torch.distributed.all_gather_into_tensor(full_output, local_input, group=group),
        )

    def _submit_fs_chunk(
        self,
        *,
        source: torch.Tensor,
        local_input: torch.Tensor,
        full_output: torch.Tensor,
        chunk_meta: ChunkMeta,
        streams: TransportStreams,
        events: ChunkEvents,
        non_blocking: bool,
        trace: TraceFactory,
        collective: Callable[[], None],
    ) -> ChunkCompletion:
        """Shared H2D(chunk) -> collective(chunk) schedule for FS-sharded input."""
        if events.input_reusable is not None:
            streams.copy.wait_event(events.input_reusable)
        with current_omni_platform.stream(streams.copy):
            with trace("h2d"):
                local_input.copy_(source, non_blocking=non_blocking)
            events.h2d_done.record(streams.copy)

        streams.communication.wait_event(events.h2d_done)
        with current_omni_platform.stream(streams.communication):
            with trace(self._transport_trace_name):
                collective()
            events.transport_done.record(streams.communication)

        element_size = full_output.element_size()
        fabric_bytes = 0
        if self.capability.rank == 0:
            fabric_bytes = chunk_meta.padded_numel * element_size * (self.capability.world_size - 1)
        self._count_chunk(source.numel() * source.element_size(), fabric_bytes)
        return ChunkCompletion(events.transport_done, streams.communication)


class GroupScatterAllGatherBackend(ReferenceBackend):
    """Safe H2D plus all-gather implementation for FS-sharded Host input.

    This is intentionally the Stage-1 reference scheduling contract.  Some
    HCCL versions do not support multiple outstanding all-gathers on one
    process group when ordering is represented only with caller-stream
    events.  Keeping each collective's normal completion semantics prevents
    an input slot from being overwritten while HCCL still reads it.
    """

    kind = TransportBackendKind.GROUP_SCATTER_AG


class GroupPersistentBackend(ReferenceBackend):
    """Persistent launch state for stable FS chunk schedules.

    One stable H2D + HCCL chunk schedule is captured in an NPUGraph per
    double-buffer slot and replayed afterwards.  ``native_persistent`` is
    validated at selection time, so a capture failure is a hard error
    (design sections 15.3 and 27), not a silent runtime fallback.
    """

    kind = TransportBackendKind.GROUP_PERSISTENT
    _transport_trace_name = "persistent_replay"

    def __init__(self, capability: TransportCapability) -> None:
        super().__init__(capability)
        self._graphs: dict[tuple[Any, ...], Any] = {}

    def submit_chunk(
        self,
        *,
        source: torch.Tensor | None,
        local_input: torch.Tensor | None,
        full_output: torch.Tensor,
        chunk_meta: ChunkMeta,
        streams: TransportStreams,
        events: ChunkEvents,
        group: torch.distributed.ProcessGroup | None,
        generation: int,
        non_blocking: bool,
        trace: TraceFactory,
    ) -> ChunkCompletion:
        del generation
        if source is None or local_input is None or group is None:
            raise RuntimeError("group_persistent requires Host source, local input, and process group")

        key = (
            id(group),
            local_input.dtype,
            local_input.numel(),
            local_input.data_ptr(),
            full_output.numel(),
            full_output.data_ptr(),
            chunk_meta.padded_numel,
        )

        def collective() -> None:
            graph = self._graphs.get(key)
            if graph is None:
                if self._metrics_on:
                    self.counters.schedule_builds += 1
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=streams.communication):
                    torch.distributed.all_gather_into_tensor(full_output, local_input, group=group)
                self._graphs[key] = graph
            else:
                if self._metrics_on:
                    self.counters.schedule_replays += 1
            graph.replay()

        return self._submit_fs_chunk(
            source=source,
            local_input=local_input,
            full_output=full_output,
            chunk_meta=chunk_meta,
            streams=streams,
            events=events,
            non_blocking=non_blocking,
            trace=trace,
            collective=collective,
        )

    def close(self) -> None:
        self._graphs.clear()
        super().close()


class PairCopyBackend(_BaseBackend):
    kind = TransportBackendKind.PAIR_COPY
    source_layout = SourceLayout.PAIR_LEADER_FULL_HOST
    requires_local_input = False
    writes_output_on_copy = True

    def submit_chunk(
        self,
        *,
        source: torch.Tensor | None,
        local_input: torch.Tensor | None,
        full_output: torch.Tensor,
        chunk_meta: ChunkMeta,
        streams: TransportStreams,
        events: ChunkEvents,
        group: torch.distributed.ProcessGroup | None,
        generation: int,
        non_blocking: bool,
        trace: TraceFactory,
    ) -> ChunkCompletion:
        del local_input, generation
        if group is None:
            raise RuntimeError("pair_copy requires the FS process group")
        pair = next((pair for pair in _pair_ranks(self.capability) if self.capability.rank in pair), None)
        if pair is None:
            raise RuntimeError(f"no pair assignment for FS local rank {self.capability.rank}")
        leader, follower = pair
        is_leader = self.capability.rank == leader
        peer_rank = follower if is_leader else leader
        peer_global_rank = self.capability.global_ranks[peer_rank]
        pair_group = self.capability.pair_group
        if pair_group is None:
            raise RuntimeError("pair_copy requires a pre-created pair process group")

        if is_leader:
            if source is None or source.numel() != chunk_meta.padded_numel:
                raise RuntimeError("pair leader requires one full padded Host chunk")
            with current_omni_platform.stream(streams.copy):
                with trace("h2d"):
                    full_output.copy_(source, non_blocking=non_blocking)
                events.h2d_done.record(streams.copy)
            streams.communication.wait_event(events.h2d_done)

        # H2D for this chunk is issued without a Host wait for the previous
        # pair transfer. The communication stream preserves P2P chunk order
        # while the copy stream can execute H2D(c1) || P2P(c0). Work objects
        # are retained and retired together by finalize_part.
        with current_omni_platform.stream(streams.communication):
            with trace("pair_copy"):
                op = torch.distributed.P2POp(
                    torch.distributed.isend if is_leader else torch.distributed.irecv,
                    full_output,
                    peer_global_rank,
                    pair_group,
                )
                works = tuple(torch.distributed.batch_isend_irecv([op]))
            events.transport_done.record(streams.communication)

        chunk_bytes = full_output.numel() * full_output.element_size()
        self._count_chunk(
            chunk_bytes if is_leader else 0,
            chunk_bytes if is_leader else 0,
            p2p_hops=1 if is_leader else 0,
            async_works=len(works),
        )
        return ChunkCompletion(events.transport_done, streams.communication, works)


class GroupPipelineMemcpyBackend(_BaseBackend):
    kind = TransportBackendKind.GROUP_PIPELINE_MEMCPY
    source_layout = SourceLayout.GROUP_OWNER_FULL_HOST
    requires_local_input = False
    writes_output_on_copy = True

    def __init__(self, capability: TransportCapability) -> None:
        super().__init__(capability)
        self._hop_streams: tuple[Any, ...] | None = None

    def _ensure_hop_streams(self) -> tuple[Any, ...]:
        if self._hop_streams is None:
            self._hop_streams = tuple(current_omni_platform.Stream() for _ in range(self.capability.world_size - 1))
        return self._hop_streams

    def begin_part(self, streams: TransportStreams, prior_last_use: Any | None) -> None:
        if self._closed:
            raise RuntimeError("weight transport backend is closed")
        if self._metrics_on:
            self.counters.submitted_parts += 1
        if prior_last_use is None:
            return
        if _pipeline_ranks(self.capability).index(self.capability.rank) == 0:
            streams.copy.wait_event(prior_last_use)
        for hop_stream in self._ensure_hop_streams():
            hop_stream.wait_event(prior_last_use)

    def submit_chunk(
        self,
        *,
        source: torch.Tensor | None,
        local_input: torch.Tensor | None,
        full_output: torch.Tensor,
        chunk_meta: ChunkMeta,
        streams: TransportStreams,
        events: ChunkEvents,
        group: torch.distributed.ProcessGroup | None,
        generation: int,
        non_blocking: bool,
        trace: TraceFactory,
    ) -> ChunkCompletion:
        del local_input, generation
        if group is None:
            raise RuntimeError("group_pipeline_memcpy requires the FS process group")

        order = _pipeline_ranks(self.capability)
        rank = self.capability.rank
        order_index = order.index(rank)
        last_index = len(order) - 1
        hop_streams = self._ensure_hop_streams()
        hop_groups = self.capability.pipeline_hop_groups
        if len(hop_groups) != last_index:
            raise RuntimeError("group_pipeline_memcpy requires pre-created process groups for every hop")

        if order_index == 0:
            if source is None or source.numel() != chunk_meta.padded_numel:
                raise RuntimeError("pipeline owner requires one full padded Host chunk")
            with current_omni_platform.stream(streams.copy):
                with trace("h2d"):
                    full_output.copy_(source, non_blocking=non_blocking)
                events.h2d_done.record(streams.copy)
        else:
            incoming_hop = order_index - 1
            incoming = hop_streams[incoming_hop]
            incoming_group = hop_groups[incoming_hop]
            if incoming_group is None:
                raise RuntimeError(f"pipeline rank {rank} is missing hop {incoming_hop} process group")
            with current_omni_platform.stream(incoming):
                with trace(f"pipeline_recv_hop_{incoming_hop}"):
                    torch.distributed.broadcast(
                        full_output,
                        src=self.capability.global_ranks[order[order_index - 1]],
                        group=incoming_group,
                    )
                receive_done = events.transport_done if order_index == last_index else events.relay_done
                if receive_done is None:
                    raise RuntimeError("pipeline relay rank requires a persistent relay event")
                receive_done.record(incoming)

            if order_index == last_index:
                completion_stream = incoming

        if order_index < last_index:
            outgoing_hop = order_index
            outgoing = hop_streams[outgoing_hop]
            outgoing_group = hop_groups[outgoing_hop]
            if outgoing_group is None:
                raise RuntimeError(f"pipeline rank {rank} is missing hop {outgoing_hop} process group")
            if order_index == 0:
                outgoing.wait_event(events.h2d_done)
            elif events.relay_done is not None:
                outgoing.wait_event(events.relay_done)
            with current_omni_platform.stream(outgoing):
                with trace(f"pipeline_send_hop_{outgoing_hop}"):
                    torch.distributed.broadcast(
                        full_output,
                        src=self.capability.global_ranks[order[order_index]],
                        group=outgoing_group,
                    )
                events.transport_done.record(outgoing)
            completion_stream = outgoing

        chunk_bytes = full_output.numel() * full_output.element_size()
        self._count_chunk(
            chunk_bytes if order_index == 0 else 0,
            chunk_bytes if order_index < last_index else 0,
            p2p_hops=1 if order_index < last_index else 0,
        )
        return ChunkCompletion(events.transport_done, completion_stream)

    def close(self) -> None:
        if self._hop_streams is not None:
            for hop_stream in self._hop_streams:
                hop_stream.synchronize()
            self._hop_streams = None
        super().close()


def create_transport_backend(
    selection: TransportSelection,
    capability: TransportCapability,
) -> WeightTransportBackend:
    backend_type: type[_BaseBackend]
    if selection.effective_backend is TransportBackendKind.REFERENCE:
        backend_type = ReferenceBackend
    elif selection.effective_backend is TransportBackendKind.GROUP_SCATTER_AG:
        backend_type = GroupScatterAllGatherBackend
    elif selection.effective_backend is TransportBackendKind.GROUP_PERSISTENT:
        backend_type = GroupPersistentBackend
    elif selection.effective_backend is TransportBackendKind.PAIR_COPY:
        backend_type = PairCopyBackend
    elif selection.effective_backend is TransportBackendKind.GROUP_PIPELINE_MEMCPY:
        backend_type = GroupPipelineMemcpyBackend
    else:
        raise RuntimeError(
            f"effective transport backend {selection.effective_backend.value} has no validated implementation"
        )
    return backend_type(capability)
