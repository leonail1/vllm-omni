# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Layout and lifecycle contracts for chunked weight transport.

This module deliberately has no platform-stream dependency.  Manifest and
packing correctness can therefore be tested on CPU before a backend submits
H2D copies or collectives.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


def ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    return (value + divisor - 1) // divisor


def round_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def dtype_element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


class ParameterOwner(str, Enum):
    FSDP_MANAGED = "fsdp_managed"
    CHUNKED_FS_OFFLOAD = "chunked_fs_offload"
    RESIDENT = "resident"
    UNSUPPORTED_RESIDENT = "unsupported_resident"


class WeightLayout(str, Enum):
    CHUNK_MAJOR = "chunk_major"
    WHOLE_BLOCK = "whole_block"


class SourceLayout(str, Enum):
    FS_SHARDED_HOST = "fs_sharded_host"
    PAIR_LEADER_FULL_HOST = "pair_leader_full_host"
    GROUP_OWNER_FULL_HOST = "group_owner_full_host"


class TransportBackendKind(str, Enum):
    AUTO = "auto"
    REFERENCE = "reference"
    PAIR_COPY = "pair_copy"
    GROUP_PERSISTENT = "group_persistent"
    GROUP_SCATTER_AG = "group_scatter_ag"
    GROUP_PIPELINE_MEMCPY = "group_pipeline_memcpy"


class PinFailurePolicy(str, Enum):
    FAIL = "fail"
    WHOLE_BLOCK_FALLBACK = "whole_block_fallback"


class SlotPhase(str, Enum):
    """Output-slot lifecycle phases (design section 25).

    Main path:

        EMPTY -> PREFETCH_ENQUEUED -> READY -> BOUND -> IN_USE -> RETIRED -> REUSABLE

    Failure paths:

        pre-collective failure   -> CANCELLED              (section 27.1)
        post-collective failure  -> POISONED_PROCESS_GROUP (section 27.2)
    """

    EMPTY = "empty"
    PREFETCH_ENQUEUED = "prefetch_enqueued"
    READY = "ready"
    BOUND = "bound"
    IN_USE = "in_use"
    RETIRED = "retired"
    REUSABLE = "reusable"
    CANCELLED = "cancelled"
    POISONED_PROCESS_GROUP = "poisoned_process_group"


# Phases in which a slot is occupied by a live transfer: the ticket is
# current and the slot cannot take a different collective key.
_ACTIVE_SLOT_PHASES = frozenset(
    {
        SlotPhase.PREFETCH_ENQUEUED,
        SlotPhase.READY,
        SlotPhase.BOUND,
        SlotPhase.IN_USE,
    }
)


@dataclass(frozen=True)
class TensorMeta:
    name: str
    offset: int
    numel: int
    shape: tuple[int, ...]
    is_buffer: bool = False
    owner: ParameterOwner = ParameterOwner.CHUNKED_FS_OFFLOAD
    # Physical stride of the source tensor; ``numel`` counts *storage*
    # elements so non-contiguous views (e.g. online FP8 Cutlass transposed
    # weights) round-trip through the flat transport buffer without changing
    # their physical layout.  None means a legacy contiguous manifest entry.
    stride: tuple[int, ...] | None = None


# Stage-3 part-pipeline contracts (design sections 20-23).
#
# A model declares the compute-order weight use of each streamed block with a
# ``BlockWeightUsePlan`` template on the block class
# (``_block_weight_use_plan``).  The engine binds the runtime block id when it
# builds per-part manifests; the template deliberately carries no block id
# because the declaring model class cannot know it.
#
# The first version supports exactly two parts per block whose lifetimes do
# not overlap: an attention part used between ``block_start`` and
# ``mid_block``, and an MoE/FFN part used between ``mid_block`` and
# ``block_end``.  A parameter read in both phases must be declared resident
# (kept on device, outside the chunk transport) — it must NOT be packed into
# both parts, which would double Host/HBM/collective bytes.
PART_ID_ATTENTION = "attention"
PART_ID_MOE = "moe"

BOUNDARY_BLOCK_START = "block_start"
BOUNDARY_MID_BLOCK = "mid_block"
BOUNDARY_BLOCK_END = "block_end"


@dataclass(frozen=True)
class WeightPartSpec:
    """One compute-phase weight partition of a streamed block.

    Attributes:
        part_id: Stable identifier used in manifests, tickets and collective
            keys (e.g. ``attention`` / ``moe``).
        module_paths: Block-relative module paths whose parameters/buffers
            belong to this part (e.g. ``("input_layernorm", "cross_attention")``).
        first_use_boundary: Forward boundary where the part is first read.
        last_use_boundary: Forward boundary after which the part is never read
            again in this block's forward.
        allowed_dtypes: Dtypes this part may carry through the chunk
            transport; anything else is rejected at planning time.
    """

    part_id: str
    module_paths: tuple[str, ...]
    first_use_boundary: str
    last_use_boundary: str
    allowed_dtypes: tuple[torch.dtype, ...] = ()


@dataclass(frozen=True)
class BlockWeightUsePlan:
    """Model-declared compute-aware weight use plan for one block type.

    Declared as ``_block_weight_use_plan`` on the block module class.  The
    engine resolves it against each concrete block instance at planning time
    and binds the runtime block id into the resulting part manifests.
    """

    parts: tuple[WeightPartSpec, ...]

    def part(self, part_id: str) -> WeightPartSpec:
        for spec in self.parts:
            if spec.part_id == part_id:
                return spec
        raise KeyError(f"weight use plan has no part {part_id!r}")


@dataclass(frozen=True)
class ChunkMeta:
    chunk_id: int
    cpu_offset: int
    full_offset: int
    valid_numel: int
    padded_numel: int
    local_numel: int


@dataclass(frozen=True)
class DTypeManifest:
    dtype: torch.dtype
    tensors: tuple[TensorMeta, ...]
    chunks: tuple[ChunkMeta, ...]
    total_numel: int
    padded_numel: int
    local_numel: int
    local_chunk_numel: int
    alignment_numel: int

    @property
    def pinned_bytes(self) -> int:
        return self.local_numel * dtype_element_size(self.dtype)


@dataclass(frozen=True)
class PartManifest:
    block_id: int
    part_id: str
    weight_shard_size: int
    weight_shard_rank: int
    chunk_size_bytes: int
    alignment_bytes: int
    layout: WeightLayout
    source_layout: SourceLayout
    dtypes: tuple[DTypeManifest, ...]
    digest: str

    @property
    def pinned_bytes(self) -> int:
        return sum(
            self.source_numel(dtype_manifest) * dtype_element_size(dtype_manifest.dtype)
            for dtype_manifest in self.dtypes
        )

    @property
    def chunk_count(self) -> int:
        return sum(len(dtype_manifest.chunks) for dtype_manifest in self.dtypes)

    def source_numel(self, dtype_manifest: DTypeManifest) -> int:
        if self.source_layout is SourceLayout.FS_SHARDED_HOST:
            return dtype_manifest.local_numel
        if self.source_layout is SourceLayout.PAIR_LEADER_FULL_HOST:
            return dtype_manifest.padded_numel if self.weight_shard_rank % 2 == 0 else 0
        if self.source_layout is SourceLayout.GROUP_OWNER_FULL_HOST:
            return dtype_manifest.padded_numel if self.weight_shard_rank == 0 else 0
        raise AssertionError(f"unhandled source layout: {self.source_layout}")


@dataclass
class PinBudget:
    """Exact pinned-memory reservation gate shared by one engine."""

    limit_bytes: int | None
    required_bytes: int = 0
    reserved_bytes: int = 0
    allocations: dict[str, int] = field(default_factory=dict)

    def plan(self, key: str, size_bytes: int) -> None:
        if size_bytes < 0:
            raise ValueError(f"pin size must be non-negative, got {size_bytes}")
        if key in self.allocations:
            raise ValueError(f"duplicate pin budget key: {key}")
        next_required = self.required_bytes + size_bytes
        if self.limit_bytes is not None and next_required > self.limit_bytes:
            raise MemoryError(
                f"pinned Host budget exceeded: required={next_required} limit={self.limit_bytes} key={key}"
            )
        self.allocations[key] = size_bytes
        self.required_bytes = next_required

    def reserve(self, key: str) -> None:
        try:
            size_bytes = self.allocations[key]
        except KeyError as exc:
            raise KeyError(f"pin allocation was not planned: {key}") from exc
        self.reserved_bytes += size_bytes
        if self.reserved_bytes > self.required_bytes:
            raise RuntimeError("pinned Host reservation exceeds the planned budget")


TensorSpec = tuple[str, torch.Tensor, bool]
PinnedAllocator = Callable[[int, torch.dtype], torch.Tensor]


def is_chunk_transport_supported(tensor: torch.Tensor) -> bool:
    return tensor.ndim > 0 and (tensor.is_floating_point() or tensor.is_complex())


# ---------------------------------------------------------------------- #
#  Metrics gate                                                           #
# ---------------------------------------------------------------------- #
#
# Transport activity counters exist only for the metrics RPC.  They are
# maintained only when VLLM_OMNI_DLO_METRICS is set at process start; with
# the default (off) the hot path pays one cached boolean branch per
# transfer instead of counter updates nobody reads.

_DLO_METRICS_ENV = "VLLM_OMNI_DLO_METRICS"
_metrics_enabled: bool | None = None


def metrics_enabled() -> bool:
    """Whether transport activity counters are maintained (read once, cached)."""
    global _metrics_enabled
    if _metrics_enabled is None:
        _metrics_enabled = os.environ.get(_DLO_METRICS_ENV, "").lower() in ("1", "true", "yes", "on")
    return _metrics_enabled


def _set_metrics_enabled_for_tests(value: bool | None) -> None:
    """Override the metrics gate; None re-reads the environment on next use."""
    global _metrics_enabled
    _metrics_enabled = value


# ---------------------------------------------------------------------- #
#  Process-group poison registry (design section 27.2)                   #
# ---------------------------------------------------------------------- #
#
# A mid-collective failure makes the whole FS process group untrustworthy.
# The slot state machine blocks further operations locally; this registry
# additionally exposes the poison to the hosting worker process so the RPC
# layer can fail closed (fatal engine failure) instead of letting every
# later request trip over the poisoned state one by one.

_POISON_LOCK = threading.Lock()
_POISON_REASON: str | None = None

# Marker embedded in worker RPC error strings when the root cause is a
# poisoned process group; the executor fail-closes on it.
DLO_POISON_ERROR_MARKER = "DLO_PROCESS_GROUP_POISONED"


def is_dlo_poison_error(error_text: Any) -> bool:
    """Whether an RPC error string reports a poisoned FS process group."""
    return isinstance(error_text, str) and DLO_POISON_ERROR_MARKER in error_text


def record_process_group_poison(reason: str) -> None:
    global _POISON_REASON
    with _POISON_LOCK:
        if _POISON_REASON is None:
            _POISON_REASON = reason


def process_group_poison_reason() -> str | None:
    with _POISON_LOCK:
        return _POISON_REASON


def _clear_process_group_poison_for_tests() -> None:
    global _POISON_REASON
    with _POISON_LOCK:
        _POISON_REASON = None


def _full_chunk_numel(
    dtype: torch.dtype,
    weight_shard_size: int,
    chunk_size_bytes: int,
    alignment_bytes: int,
) -> tuple[int, int]:
    if weight_shard_size <= 0:
        raise ValueError(f"weight_shard_size must be positive, got {weight_shard_size}")
    if chunk_size_bytes <= 0:
        raise ValueError(f"chunk_size_bytes must be positive, got {chunk_size_bytes}")
    if alignment_bytes <= 0:
        raise ValueError(f"alignment_bytes must be positive, got {alignment_bytes}")
    element_size = dtype_element_size(dtype)
    alignment_numel = ceil_div(alignment_bytes, element_size)
    collective_alignment = weight_shard_size * alignment_numel
    requested_numel = chunk_size_bytes // element_size
    full_chunk_numel = requested_numel - requested_numel % collective_alignment
    if full_chunk_numel == 0:
        raise ValueError(
            "chunk size is smaller than one aligned collective unit: "
            f"chunk_size_bytes={chunk_size_bytes}, dtype={dtype}, "
            f"weight_shard_size={weight_shard_size}, alignment_bytes={alignment_bytes}"
        )
    return full_chunk_numel, alignment_numel


def build_part_manifest(
    tensor_specs: Sequence[TensorSpec],
    *,
    block_id: int,
    part_id: str,
    weight_shard_size: int,
    weight_shard_rank: int,
    chunk_size_bytes: int,
    alignment_bytes: int = 256,
    layout: WeightLayout = WeightLayout.CHUNK_MAJOR,
    source_layout: SourceLayout = SourceLayout.FS_SHARDED_HOST,
) -> PartManifest:
    if not 0 <= weight_shard_rank < weight_shard_size:
        raise ValueError(f"weight_shard_rank={weight_shard_rank} is outside [0, {weight_shard_size})")

    grouped: OrderedDict[torch.dtype, list[TensorSpec]] = OrderedDict()
    for name, tensor, is_buffer in tensor_specs:
        if not is_chunk_transport_supported(tensor):
            continue
        grouped.setdefault(tensor.dtype, []).append((name, tensor, is_buffer))

    dtype_manifests: list[DTypeManifest] = []
    for dtype, dtype_specs in grouped.items():
        offset = 0
        tensor_metas: list[TensorMeta] = []
        for name, tensor, is_buffer in dtype_specs:
            stride = tensor.stride()
            storage_numel = (
                0
                if tensor.numel() == 0
                else 1 + sum((size - 1) * axis_stride for size, axis_stride in zip(tensor.shape, stride))
            )
            tensor_metas.append(
                TensorMeta(
                    name=name,
                    offset=offset,
                    numel=storage_numel,
                    shape=tuple(tensor.shape),
                    is_buffer=is_buffer,
                    stride=tuple(stride),
                )
            )
            offset += storage_numel

        total_numel = offset
        chunks: list[ChunkMeta] = []
        cpu_offset = 0
        if layout is WeightLayout.CHUNK_MAJOR:
            full_chunk_numel, alignment_numel = _full_chunk_numel(
                dtype,
                weight_shard_size,
                chunk_size_bytes,
                alignment_bytes,
            )
            for chunk_id, full_offset in enumerate(range(0, total_numel, full_chunk_numel)):
                valid_numel = min(full_chunk_numel, total_numel - full_offset)
                padded_numel = round_up(
                    valid_numel,
                    weight_shard_size * alignment_numel,
                )
                local_numel = padded_numel // weight_shard_size
                chunks.append(
                    ChunkMeta(
                        chunk_id=chunk_id,
                        cpu_offset=cpu_offset,
                        full_offset=full_offset,
                        valid_numel=valid_numel,
                        padded_numel=padded_numel,
                        local_numel=local_numel,
                    )
                )
                cpu_offset += local_numel
        else:
            alignment_numel = 1
            local_numel = ceil_div(total_numel, weight_shard_size)
            padded_numel = local_numel * weight_shard_size
            chunks.append(
                ChunkMeta(
                    chunk_id=0,
                    cpu_offset=0,
                    full_offset=0,
                    valid_numel=total_numel,
                    padded_numel=padded_numel,
                    local_numel=local_numel,
                )
            )
            cpu_offset = local_numel

        padded_total = chunks[-1].full_offset + chunks[-1].padded_numel if chunks else 0
        dtype_manifests.append(
            DTypeManifest(
                dtype=dtype,
                tensors=tuple(tensor_metas),
                chunks=tuple(chunks),
                total_numel=total_numel,
                padded_numel=padded_total,
                local_numel=cpu_offset,
                local_chunk_numel=max((chunk.local_numel for chunk in chunks), default=0),
                alignment_numel=alignment_numel,
            )
        )

    digest_payload = {
        "block_id": block_id,
        "part_id": part_id,
        "weight_shard_size": weight_shard_size,
        "chunk_size_bytes": chunk_size_bytes,
        "alignment_bytes": alignment_bytes,
        "layout": layout.value,
        "source_layout": source_layout.value,
        "dtypes": [
            {
                "dtype": str(dtype_manifest.dtype),
                "tensors": [
                    {
                        "name": tensor.name,
                        "offset": tensor.offset,
                        "numel": tensor.numel,
                        "shape": tensor.shape,
                        "is_buffer": tensor.is_buffer,
                        "owner": tensor.owner.value,
                        "stride": tensor.stride,
                    }
                    for tensor in dtype_manifest.tensors
                ],
                "chunks": [chunk.__dict__ for chunk in dtype_manifest.chunks],
            }
            for dtype_manifest in dtype_manifests
        ],
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return PartManifest(
        block_id=block_id,
        part_id=part_id,
        weight_shard_size=weight_shard_size,
        weight_shard_rank=weight_shard_rank,
        chunk_size_bytes=chunk_size_bytes,
        alignment_bytes=alignment_bytes,
        layout=layout,
        source_layout=source_layout,
        dtypes=tuple(dtype_manifests),
        digest=digest,
    )


def _default_pinned_allocator(numel: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)


def pack_local_shard(
    tensor_specs: Sequence[TensorSpec],
    manifest: PartManifest,
    *,
    allocator: PinnedAllocator | None = None,
) -> dict[torch.dtype, torch.Tensor]:
    """Pack final tensors directly into the manifest's local Host layout."""
    allocator = allocator or _default_pinned_allocator
    sources = {name: tensor for name, tensor, _ in tensor_specs}
    packed: dict[torch.dtype, torch.Tensor] = {}

    for dtype_manifest in manifest.dtypes:
        source_numel = manifest.source_numel(dtype_manifest)
        local = allocator(source_numel, dtype_manifest.dtype)
        if local.device.type != "cpu":
            raise ValueError(f"pinned shard allocator returned non-CPU tensor: {local.device}")
        local.zero_()

        if source_numel == 0:
            packed[dtype_manifest.dtype] = local
            continue

        if manifest.source_layout is not SourceLayout.FS_SHARDED_HOST:
            _copy_flat_range(
                local,
                dst_offset=0,
                source_begin=0,
                source_end=dtype_manifest.total_numel,
                tensor_metas=dtype_manifest.tensors,
                sources=sources,
            )
            packed[dtype_manifest.dtype] = local
            continue

        if manifest.layout is WeightLayout.WHOLE_BLOCK:
            chunk = dtype_manifest.chunks[0]
            shard_begin = manifest.weight_shard_rank * chunk.local_numel
            shard_end = min(shard_begin + chunk.local_numel, dtype_manifest.total_numel)
            _copy_flat_range(
                local,
                dst_offset=0,
                source_begin=shard_begin,
                source_end=shard_end,
                tensor_metas=dtype_manifest.tensors,
                sources=sources,
            )
        else:
            for chunk in dtype_manifest.chunks:
                source_begin = chunk.full_offset + manifest.weight_shard_rank * chunk.local_numel
                source_end = min(
                    source_begin + chunk.local_numel,
                    chunk.full_offset + chunk.valid_numel,
                )
                _copy_flat_range(
                    local,
                    dst_offset=chunk.cpu_offset,
                    source_begin=source_begin,
                    source_end=source_end,
                    tensor_metas=dtype_manifest.tensors,
                    sources=sources,
                )
        packed[dtype_manifest.dtype] = local

    return packed


def _flat_physical(source: torch.Tensor, storage_numel: int) -> torch.Tensor:
    """Flatten *source* in physical storage order.

    Contiguous tensors flatten logically (which is also their physical
    order).  Non-contiguous views — online FP8 stores Cutlass weights as
    transposed views (e.g. stride=(1, K)) — are copied through a physical
    layout so the flat transport buffer round-trips with the original
    stride; flattening them logically and rebuilding with ``.view()`` would
    change the layout and make scaled_mm reject the weight.
    """
    if source.is_contiguous():
        return source.reshape(-1)
    flat = torch.empty(storage_numel, dtype=source.dtype, device=source.device)
    torch.as_strided(flat, size=source.shape, stride=source.stride()).copy_(source)
    return flat


def _copy_flat_range(
    destination: torch.Tensor,
    *,
    dst_offset: int,
    source_begin: int,
    source_end: int,
    tensor_metas: Iterable[TensorMeta],
    sources: dict[str, torch.Tensor],
) -> None:
    if source_end <= source_begin:
        return
    for tensor_meta in tensor_metas:
        tensor_begin = tensor_meta.offset
        tensor_end = tensor_begin + tensor_meta.numel
        overlap_begin = max(source_begin, tensor_begin)
        overlap_end = min(source_end, tensor_end)
        if overlap_begin >= overlap_end:
            continue
        source = _flat_physical(sources[tensor_meta.name], tensor_meta.numel)
        source_offset = overlap_begin - tensor_begin
        count = overlap_end - overlap_begin
        destination_offset = dst_offset + overlap_begin - source_begin
        destination[destination_offset : destination_offset + count].copy_(
            source[source_offset : source_offset + count]
        )


def reconstruct_full_flat(
    local_shards: Sequence[torch.Tensor],
    dtype_manifest: DTypeManifest,
    *,
    layout: WeightLayout = WeightLayout.CHUNK_MAJOR,
) -> torch.Tensor:
    """CPU reference reconstruction for synchronous parity tests."""
    if not local_shards:
        raise ValueError("at least one local shard is required")
    full = torch.empty(dtype_manifest.total_numel, dtype=dtype_manifest.dtype)
    if layout is WeightLayout.WHOLE_BLOCK:
        gathered = torch.cat(local_shards)
        full.copy_(gathered[: dtype_manifest.total_numel])
        return full

    for chunk in dtype_manifest.chunks:
        gathered = torch.cat([local[chunk.cpu_offset : chunk.cpu_offset + chunk.local_numel] for local in local_shards])
        full[chunk.full_offset : chunk.full_offset + chunk.valid_numel].copy_(gathered[: chunk.valid_numel])
    return full


@dataclass(frozen=True)
class TransferTicket:
    request_generation: int
    forward_generation: int
    block_id: int
    part_id: str
    output_slot: int
    chunk_count: int
    ready_event: Any | None = field(default=None, compare=False)
    last_collective_key: tuple[Any, ...] | None = None
    backend_id: str | None = None

    @property
    def owner_key(self) -> tuple[int, int, int, str]:
        return (
            self.request_generation,
            self.forward_generation,
            self.block_id,
            self.part_id,
        )

    @property
    def validation_key(self) -> tuple[Any, ...]:
        """The section-25 transition checklist captured at ``begin`` time.

        Carries generation, block/part, backend, output slot and the expected
        collective key (section 24); every later transition recomputes this
        from the presented ticket and compares it against the slot's record.
        """
        return (
            self.request_generation,
            self.forward_generation,
            self.block_id,
            self.part_id,
            self.output_slot,
            self.backend_id,
            self.last_collective_key,
        )


@dataclass
class OutputSlotState:
    phase: SlotPhase = SlotPhase.EMPTY
    ticket: TransferTicket | None = None
    expected_key: tuple[Any, ...] | None = None
    last_use_event: Any | None = None
    fallback_reason: str | None = None
    error: str | None = None


class ChunkTransportState:
    """Own generation, key and conflict checks for persistent output slots.

    Implements the design section 25 slot state machine:

        EMPTY -> PREFETCH_ENQUEUED -> READY -> BOUND -> IN_USE -> RETIRED -> REUSABLE

    with the two failure exits CANCELLED (pre-collective, section 27.1) and
    POISONED_PROCESS_GROUP (post-collective, section 27.2).  Every transition
    revalidates the section-25 checklist against the key recorded at
    ``begin``: generation, block/part, backend, output slot, prior last-use
    and the expected collective key.
    """

    def __init__(self, block_id: int, slot_count: int = 2) -> None:
        self.block_id = block_id
        self._forward_generation = 0
        self._slots = [OutputSlotState() for _ in range(slot_count)]
        self._closed = False
        self._poisoned: str | None = None

    @property
    def slots(self) -> tuple[OutputSlotState, ...]:
        return tuple(self._slots)

    @property
    def poisoned(self) -> str | None:
        """Why the process group was poisoned, or None while operational."""
        return self._poisoned

    def _require_operational(self) -> None:
        if self._closed:
            raise RuntimeError("chunk transport state is closed")
        if self._poisoned is not None:
            raise RuntimeError(
                f"process group is poisoned ({self._poisoned}); the affected "
                "worker/process group must be rebuilt, not reused (design section 27.2)"
            )

    def begin(
        self,
        *,
        output_slot: int,
        chunk_count: int,
        request_generation: int = 0,
        part_id: str = "block",
        ready_event: Any | None = None,
        backend_id: str | None = None,
        last_collective_key: tuple[Any, ...] | None = None,
    ) -> TransferTicket:
        """Enqueue a new transfer into *output_slot*: -> PREFETCH_ENQUEUED.

        A retired slot must first prove its consumer's last-use before being
        reused.  Re-submitting the identical collective key onto a live slot
        returns the existing ticket; a different key fails fast (section 25).
        """
        self._require_operational()
        slot = self._slots[output_slot]

        if slot.phase is SlotPhase.RETIRED:
            # Prior last-use check (sections 4.5/25): record_last_use only
            # lands here after a last-use (or overwrite) event was recorded,
            # so the producer may safely reuse the slot.
            slot.phase = SlotPhase.REUSABLE

        if slot.phase in _ACTIVE_SLOT_PHASES:
            current = slot.ticket
            if (
                current is not None
                and current.request_generation == request_generation
                and current.block_id == self.block_id
                and current.part_id == part_id
                and current.chunk_count == chunk_count
                and current.backend_id == backend_id
                and current.last_collective_key == last_collective_key
            ):
                # Idempotent re-prefetch with the same collective key.
                return current
            raise RuntimeError(
                f"output slot {output_slot} is owned by {current.owner_key if current else slot.phase.value}: "
                "a different collective key cannot take a live slot (design section 25)"
            )

        self._forward_generation += 1
        ticket = TransferTicket(
            request_generation=request_generation,
            forward_generation=self._forward_generation,
            block_id=self.block_id,
            part_id=part_id,
            output_slot=output_slot,
            chunk_count=chunk_count,
            ready_event=ready_event,
            last_collective_key=last_collective_key,
            backend_id=backend_id,
        )
        slot.phase = SlotPhase.PREFETCH_ENQUEUED
        slot.ticket = ticket
        slot.expected_key = ticket.validation_key
        slot.error = None
        return ticket

    def mark_ready(self, ticket: TransferTicket) -> None:
        """Producer finished publishing: PREFETCH_ENQUEUED -> READY."""
        slot = self._require_current(ticket)
        if slot.phase is not SlotPhase.PREFETCH_ENQUEUED:
            raise RuntimeError(f"cannot mark {slot.phase.value} slot ready")
        slot.phase = SlotPhase.READY

    def mark_bound(self, ticket: TransferTicket) -> None:
        """Consumer Parameters were re-pointed at the slot: READY -> BOUND.

        Idempotent while the slot stays BOUND: re-pointing the same
        Parameters at the same storage is a no-op.
        """
        slot = self._require_current(ticket)
        if slot.phase is SlotPhase.BOUND:
            return
        if slot.phase is not SlotPhase.READY:
            raise RuntimeError(f"cannot bind {slot.phase.value} slot")
        slot.phase = SlotPhase.BOUND

    def mark_in_use(self, ticket: TransferTicket) -> None:
        """Compute is about to read the slot: BOUND -> IN_USE.

        A bind must have happened first: attaching a consumer straight from
        READY would read storage the Parameters do not point at yet.
        Re-attaching an IN_USE slot is allowed (same consumer contract).
        """
        slot = self._require_current(ticket)
        if slot.phase not in (SlotPhase.BOUND, SlotPhase.IN_USE):
            raise RuntimeError(f"cannot attach consumer to {slot.phase.value} slot")
        slot.phase = SlotPhase.IN_USE

    def record_last_use(self, ticket: TransferTicket, last_use_event: Any | None) -> None:
        """Consumer's last read is captured: READY/BOUND/IN_USE -> RETIRED.

        A ticket that reached IN_USE was really consumed and must carry a real
        last-use event (consumer-last-use, section 4.5).  A READY/BOUND ticket
        that was never consumed (e.g. a tail prefetch retired at
        producer-ready, or a contaminated slot overwritten before use) may
        retire with the overwrite/ready event or None.
        """
        slot = self._require_current(ticket)
        if slot.phase not in (SlotPhase.READY, SlotPhase.BOUND, SlotPhase.IN_USE):
            raise RuntimeError(f"cannot retire {slot.phase.value} slot")
        if slot.phase is SlotPhase.IN_USE and last_use_event is None:
            raise RuntimeError(
                f"cannot retire in_use slot {ticket.output_slot} without a last-use event: "
                "the next producer could overwrite weights still being read (design section 4.5)"
            )
        slot.last_use_event = last_use_event
        slot.phase = SlotPhase.RETIRED

    def confirm_reusable(self, ticket: TransferTicket) -> None:
        """Retirement is complete and the slot may be reused: RETIRED -> REUSABLE."""
        slot = self._require_current(ticket)
        if slot.phase is not SlotPhase.RETIRED:
            raise RuntimeError(f"cannot confirm reusable on {slot.phase.value} slot")
        slot.phase = SlotPhase.REUSABLE

    def cancel(self, ticket: TransferTicket, reason: str) -> None:
        """Abort before the first collective was submitted: PREFETCH_ENQUEUED -> CANCELLED.

        Only failures listed in design section 27.1 may take this exit; once
        any collective was submitted the process group can no longer be
        trusted and ``poison`` must be used instead.
        """
        slot = self._require_current(ticket)
        if slot.phase is not SlotPhase.PREFETCH_ENQUEUED:
            raise RuntimeError(
                f"cannot cancel {slot.phase.value} slot: collectives may already be submitted; "
                "use poison() instead (design section 27.2)"
            )
        slot.error = reason
        slot.phase = SlotPhase.CANCELLED

    def poison(self, reason: str) -> None:
        """A collective failed mid-flight: -> POISONED_PROCESS_GROUP.

        Marks every live slot poisoned and blocks all further state
        operations; the process group must be rebuilt (design section 27.2).
        The poison is also recorded process-wide so the worker can convert
        the next RPC failure into a fatal engine failure (fail-closed).
        """
        self._poisoned = reason
        record_process_group_poison(reason)
        for slot in self._slots:
            if slot.phase in _ACTIVE_SLOT_PHASES:
                slot.error = reason
                slot.phase = SlotPhase.POISONED_PROCESS_GROUP

    def is_current(self, ticket: TransferTicket) -> bool:
        slot = self._slots[ticket.output_slot]
        return slot.ticket == ticket and slot.phase in _ACTIVE_SLOT_PHASES

    def current_phase(self, ticket: TransferTicket) -> SlotPhase | None:
        """Phase of *ticket*'s slot if the ticket still owns it, else None."""
        slot = self._slots[ticket.output_slot]
        if slot.ticket != ticket:
            return None
        return slot.phase

    def reset(self) -> None:
        if self._poisoned is not None:
            raise RuntimeError(
                f"cannot reset a poisoned chunk transport ({self._poisoned}); "
                "rebuild the process group instead (design section 27.2)"
            )
        if any(slot.phase in _ACTIVE_SLOT_PHASES for slot in self._slots):
            raise RuntimeError("cannot reset chunk transport with in-flight slots")
        self._slots = [OutputSlotState() for _ in self._slots]
        self._forward_generation = 0
        self._closed = False

    def close(self) -> None:
        if self._poisoned is not None:
            raise RuntimeError(
                f"cannot close a poisoned chunk transport ({self._poisoned}); "
                "rebuild the process group instead (design section 27.2)"
            )
        if any(slot.phase in _ACTIVE_SLOT_PHASES for slot in self._slots):
            raise RuntimeError("cannot close chunk transport with in-flight slots")
        self._closed = True

    def _require_current(self, ticket: TransferTicket) -> OutputSlotState:
        self._require_operational()
        slot = self._slots[ticket.output_slot]
        if slot.ticket != ticket:
            raise RuntimeError(f"stale transfer ticket for output slot {ticket.output_slot}")
        if slot.expected_key != ticket.validation_key:
            raise RuntimeError(
                f"collective key mismatch on output slot {ticket.output_slot}: "
                f"slot expects {slot.expected_key}, ticket carries {ticket.validation_key} "
                "(design sections 24-25)"
            )
        return slot


@dataclass
class TransportCounters:
    submissions: int = 0
    submitted_chunks: int = 0
    consumer_attaches: int = 0
    releases: int = 0
    cancels: int = 0
    poisons: int = 0
    resets: int = 0


class ChunkedWeightTransport:
    """Reference lifecycle contract shared by platform-specific submitters.

    The backend owns stream operations; this class owns prepared Host storage,
    generation tickets, consumer attachment, last-use release, and bounded
    reset/close behavior.
    """

    def __init__(self, block_id: int, slot_count: int = 2) -> None:
        self.state = ChunkTransportState(block_id, slot_count=slot_count)
        self.manifest: PartManifest | None = None
        self.host_shards: dict[torch.dtype, torch.Tensor] = {}
        self.counters = TransportCounters()
        # Cached at construction: counter updates cost one predictable
        # branch per transfer when the metrics gate is off.
        self._metrics_on = metrics_enabled()

    def prepare(
        self,
        manifest: PartManifest,
        host_shards: dict[torch.dtype, torch.Tensor],
    ) -> None:
        if manifest.block_id != self.state.block_id:
            raise ValueError(
                f"manifest block_id={manifest.block_id} does not match transport block_id={self.state.block_id}"
            )
        expected = {dtype_manifest.dtype: manifest.source_numel(dtype_manifest) for dtype_manifest in manifest.dtypes}
        actual = {dtype: shard.numel() for dtype, shard in host_shards.items()}
        if actual != expected:
            raise ValueError(f"Host shard sizes do not match manifest: expected={expected}, actual={actual}")
        self.manifest = manifest
        self.host_shards = host_shards

    def begin_submission(
        self,
        *,
        output_slot: int,
        request_generation: int,
        ready_event: Any,
        part_id: str = "block",
        backend_id: str | None = None,
        last_collective_key: tuple[Any, ...] | None = None,
    ) -> TransferTicket:
        if self.manifest is None:
            raise RuntimeError("chunk transport is not prepared")
        previous = self.state.slots[output_slot].ticket
        ticket = self.state.begin(
            output_slot=output_slot,
            chunk_count=self.manifest.chunk_count,
            request_generation=request_generation,
            part_id=part_id,
            ready_event=ready_event,
            backend_id=backend_id,
            last_collective_key=last_collective_key,
        )
        if ticket is not previous and self._metrics_on:
            self.counters.submissions += 1
            self.counters.submitted_chunks += ticket.chunk_count
        return ticket

    def mark_ready(self, ticket: TransferTicket) -> None:
        self.state.mark_ready(ticket)

    def mark_bound(self, ticket: TransferTicket) -> None:
        self.state.mark_bound(ticket)

    def attach_ready(
        self,
        ticket: TransferTicket,
        wait_event: Callable[[Any], None],
    ) -> None:
        self.state.mark_in_use(ticket)
        wait_event(ticket.ready_event)
        if self._metrics_on:
            self.counters.consumer_attaches += 1

    def record_last_use(self, ticket: TransferTicket, last_use_event: Any) -> None:
        self.state.record_last_use(ticket, last_use_event)
        if self._metrics_on:
            self.counters.releases += 1

    def confirm_reusable(self, ticket: TransferTicket) -> None:
        self.state.confirm_reusable(ticket)

    def cancel_submission(self, ticket: TransferTicket, reason: str) -> None:
        """Abort a submission before any collective was issued (section 27.1)."""
        self.state.cancel(ticket, reason)
        if self._metrics_on:
            self.counters.cancels += 1

    def poison(self, reason: str) -> None:
        """Mark the process group untrusted after a mid-collective failure (section 27.2)."""
        self.state.poison(reason)
        if self._metrics_on:
            self.counters.poisons += 1

    def reset(self) -> None:
        self.state.reset()
        if self._metrics_on:
            self.counters.resets += 1

    def reset_counters(self) -> None:
        """Start a fresh accounting window without changing transport state."""
        self.counters = TransportCounters()

    def close(self) -> None:
        self.state.close()
        self.host_shards = {}
