# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host packing and pinned-memory budget (called at enable(), outside the submit loop)."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import torch

from ._chunk_types import DTypeManifest, PartManifest, TensorMeta, TensorSpec, WeightLayout


@dataclass
class PinBudget:
    limit_bytes: int | None
    required_bytes: int = 0
    reserved_bytes: int = 0
    allocations: dict[str, int] = field(default_factory=dict)

    def plan(self, key: str, size_bytes: int) -> None:
        if size_bytes < 0:
            raise ValueError(f"pin size must be non-negative, got {size_bytes}")
        if key in self.allocations:
            raise ValueError(f"duplicate pin budget key: {key}")
        nxt = self.required_bytes + size_bytes
        if self.limit_bytes is not None and nxt > self.limit_bytes:
            raise MemoryError(f"pinned Host budget exceeded: required={nxt} limit={self.limit_bytes} key={key}")
        self.allocations[key] = size_bytes
        self.required_bytes = nxt

    def reserve(self, key: str) -> None:
        try:
            size_bytes = self.allocations[key]
        except KeyError as exc:
            raise KeyError(f"pin allocation was not planned: {key}") from exc
        self.reserved_bytes += size_bytes
        if self.reserved_bytes > self.required_bytes:
            raise RuntimeError("pinned Host reservation exceeds the planned budget")


PinnedAllocator = Callable[[int, torch.dtype], torch.Tensor]


def _default_pinned_allocator(numel: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)


def _flat_physical(source: torch.Tensor, storage_numel: int) -> torch.Tensor:
    if source.is_contiguous():
        return source.reshape(-1)
    flat = torch.empty(storage_numel, dtype=source.dtype, device=source.device)
    torch.as_strided(flat, size=source.shape, stride=source.stride()).copy_(source)
    return flat


def _copy_flat_range(
    destination: torch.Tensor, *, dst_offset: int, source_begin: int, source_end: int,
    tensor_metas: Iterable[TensorMeta], sources: dict[str, torch.Tensor],
) -> None:
    if source_end <= source_begin:
        return
    for tensor_meta in tensor_metas:
        tensor_begin, tensor_end = tensor_meta.offset, tensor_meta.offset + tensor_meta.numel
        overlap_begin, overlap_end = max(source_begin, tensor_begin), min(source_end, tensor_end)
        if overlap_begin >= overlap_end:
            continue
        source = _flat_physical(sources[tensor_meta.name], tensor_meta.numel)
        count = overlap_end - overlap_begin
        dst = dst_offset + overlap_begin - source_begin
        destination[dst:dst + count].copy_(source[overlap_begin - tensor_begin:overlap_begin - tensor_begin + count])


def pack_local_shard(
    tensor_specs: Sequence[TensorSpec], manifest: PartManifest, *, allocator: PinnedAllocator | None = None,
) -> dict[torch.dtype, torch.Tensor]:
    allocator = allocator or _default_pinned_allocator
    sources = {name: tensor for name, tensor, _ in tensor_specs}
    packed: dict[torch.dtype, torch.Tensor] = {}
    rank = manifest.weight_shard_rank
    for dm in manifest.dtypes:
        local = allocator(dm.local_numel, dm.dtype)
        if local.device.type != "cpu":
            raise ValueError(f"pinned shard allocator returned non-CPU tensor: {local.device}")
        local.zero_()
        if manifest.layout is WeightLayout.WHOLE_BLOCK:
            chunk = dm.chunks[0]
            begin = rank * chunk.local_numel
            _copy_flat_range(
                local, dst_offset=0, source_begin=begin,
                source_end=min(begin + chunk.local_numel, dm.total_numel),
                tensor_metas=dm.tensors, sources=sources,
            )
        else:
            for chunk in dm.chunks:
                begin = chunk.full_offset + rank * chunk.local_numel
                _copy_flat_range(
                    local, dst_offset=chunk.cpu_offset, source_begin=begin,
                    source_end=min(begin + chunk.local_numel, chunk.full_offset + chunk.valid_numel),
                    tensor_metas=dm.tensors, sources=sources,
                )
        packed[dm.dtype] = local
    return packed


def reconstruct_full_flat(
    local_shards: list[torch.Tensor],
    dtype_manifest: DTypeManifest,
    *,
    layout: WeightLayout = WeightLayout.CHUNK_MAJOR,
) -> torch.Tensor:
    if not local_shards:
        raise ValueError("at least one local shard is required")
    full = torch.empty(dtype_manifest.total_numel, dtype=dtype_manifest.dtype)
    if layout is WeightLayout.WHOLE_BLOCK:
        full.copy_(torch.cat(list(local_shards))[: dtype_manifest.total_numel])
        return full
    for chunk in dtype_manifest.chunks:
        gathered = torch.cat([s[chunk.cpu_offset:chunk.cpu_offset + chunk.local_numel] for s in local_shards])
        full[chunk.full_offset:chunk.full_offset + chunk.valid_numel].copy_(gathered[: chunk.valid_numel])
    return full
