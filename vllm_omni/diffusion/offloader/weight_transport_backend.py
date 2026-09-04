# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunk data plane: Host-to-device copy, then AllGather on a multi-rank FS group."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from vllm_omni.platforms import current_omni_platform

from ._chunk_types import ChunkMeta, TransportBackendKind

TraceFactory = Callable[[str], AbstractContextManager[Any]]


@dataclass(frozen=True)
class TransportCapability:
    world_size: int
    rank: int
    global_ranks: tuple[int, ...]


@dataclass(frozen=True)
class TransportSelection:
    requested_backend: TransportBackendKind
    effective_backend: TransportBackendKind


@dataclass(frozen=True)
class TransportStreams:
    copy: Any
    communication: Any


@dataclass(frozen=True)
class ChunkEvents:
    h2d_done: Any
    transport_done: Any
    input_reusable: Any | None = None


@dataclass(frozen=True)
class ChunkCompletion:
    event: Any
    stream: Any


@dataclass
class BackendCounters:
    submitted_parts: int = 0
    submitted_chunks: int = 0
    host_h2d_bytes: int = 0
    fabric_bytes: int = 0
    backend_chunks: dict[str, int] = field(default_factory=dict)


class WeightTransportBackend(Protocol):
    kind: TransportBackendKind
    requires_local_input: bool
    counters: BackendCounters


@contextmanager
def _on_stream(stream: Any):
    if hasattr(stream, "device"):
        with current_omni_platform.stream(stream):
            yield
    else:
        yield


def select_transport(requested_backend: TransportBackendKind, capability: TransportCapability) -> TransportSelection:
    if requested_backend is TransportBackendKind.AUTO:
        candidate = (
            TransportBackendKind.GROUP_SCATTER_AG if capability.world_size > 1
            else TransportBackendKind.REFERENCE
        )
    else:
        candidate = requested_backend
    if candidate is TransportBackendKind.GROUP_SCATTER_AG and capability.world_size <= 1:
        raise ValueError(
            "transport backend=group_scatter_ag is unsupported on this host: "
            "group_scatter_ag requires an FS group larger than one rank"
        )
    if candidate not in (TransportBackendKind.REFERENCE, TransportBackendKind.GROUP_SCATTER_AG):
        raise ValueError(
            f"transport backend={candidate.value} is unsupported on this host: "
            f"unsupported backend: {candidate.value}"
        )
    return TransportSelection(requested_backend, candidate)


class ReferenceBackend:
    """Host-to-device copy; AllGather when the FS group has more than one rank."""

    kind = TransportBackendKind.REFERENCE
    requires_local_input = True
    _transport_trace_name = "all_gather"

    def __init__(self, capability: TransportCapability) -> None:
        self.capability = capability
        self.counters = BackendCounters()
        self._generation = -1
        self._closed = False
        self.requires_local_input = capability.world_size > 1
        self.writes_output_on_copy = capability.world_size <= 1

    def begin_part(self, streams: TransportStreams, prior_last_use: Any | None) -> None:
        if self._closed:
            raise RuntimeError("weight transport backend is closed")
        self.counters.submitted_parts += 1
        if prior_last_use is not None:
            streams.communication.wait_event(prior_last_use)

    def _count_chunk(self, host_bytes: int, fabric_bytes: int) -> None:
        self.counters.submitted_chunks += 1
        self.counters.host_h2d_bytes += host_bytes
        self.counters.fabric_bytes += fabric_bytes
        self.counters.backend_chunks[self.kind.value] = self.counters.backend_chunks.get(self.kind.value, 0) + 1

    def finalize_part(self, completions: Sequence[ChunkCompletion], *, ready_event: Any, streams: TransportStreams) -> Any:
        stream = completions[-1].stream if completions else streams.communication
        with _on_stream(stream):
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

    def submit_chunk(
        self, *, source: torch.Tensor | None, local_input: torch.Tensor | None, full_output: torch.Tensor,
        chunk_meta: ChunkMeta, streams: TransportStreams, events: ChunkEvents,
        group: torch.distributed.ProcessGroup | None, generation: int, non_blocking: bool, trace: TraceFactory,
    ) -> ChunkCompletion:
        del generation
        if source is None:
            raise RuntimeError("reference transport requires a local Host source")
        if self.capability.world_size <= 1:
            if source.numel() != chunk_meta.padded_numel:
                raise RuntimeError("single-rank reference transport requires one full padded Host chunk")
            with _on_stream(streams.copy):
                with trace("h2d"):
                    full_output.copy_(source, non_blocking=non_blocking)
                events.h2d_done.record(streams.copy)
            self._count_chunk(source.numel() * source.element_size(), 0)
            return ChunkCompletion(events.h2d_done, streams.copy)
        if local_input is None or group is None:
            raise RuntimeError("reference FS transport requires a local input buffer and process group")
        return self._submit_fs_chunk(
            source, local_input, full_output, chunk_meta, streams, events, non_blocking, trace,
            lambda: torch.distributed.all_gather_into_tensor(full_output, local_input, group=group),
        )

    def _submit_fs_chunk(
        self, source: torch.Tensor, local_input: torch.Tensor, full_output: torch.Tensor,
        chunk_meta: ChunkMeta, streams: TransportStreams, events: ChunkEvents,
        non_blocking: bool, trace: TraceFactory, collective: Callable[[], None],
    ) -> ChunkCompletion:
        if events.input_reusable is not None:
            streams.copy.wait_event(events.input_reusable)
        with _on_stream(streams.copy):
            with trace("h2d"):
                local_input.copy_(source, non_blocking=non_blocking)
            events.h2d_done.record(streams.copy)
        streams.communication.wait_event(events.h2d_done)
        with _on_stream(streams.communication):
            with trace(self._transport_trace_name):
                collective()
            events.transport_done.record(streams.communication)
        fabric = (
            chunk_meta.padded_numel * full_output.element_size() * (self.capability.world_size - 1)
            if self.capability.rank == 0 else 0
        )
        self._count_chunk(source.numel() * source.element_size(), fabric)
        return ChunkCompletion(events.transport_done, streams.communication)


class GroupScatterAllGatherBackend(ReferenceBackend):
    """Same Host-to-device + AllGather schedule as ReferenceBackend on a multi-rank FS group."""

    kind = TransportBackendKind.GROUP_SCATTER_AG


def create_transport_backend(selection: TransportSelection, capability: TransportCapability) -> WeightTransportBackend:
    if selection.effective_backend is TransportBackendKind.REFERENCE:
        return ReferenceBackend(capability)
    if selection.effective_backend is TransportBackendKind.GROUP_SCATTER_AG:
        return GroupScatterAllGatherBackend(capability)
    raise RuntimeError(
        f"effective transport backend {selection.effective_backend.value} has no validated implementation"
    )
