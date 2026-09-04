# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Schema types for chunked H2D + AllGather weight transport."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


def dtype_element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


class WeightLayout(str, Enum):
    CHUNK_MAJOR = "chunk_major"
    WHOLE_BLOCK = "whole_block"


class TransportBackendKind(str, Enum):
    AUTO = "auto"
    REFERENCE = "reference"
    GROUP_SCATTER_AG = "group_scatter_ag"


class PinFailurePolicy(str, Enum):
    FAIL = "fail"
    WHOLE_BLOCK_FALLBACK = "whole_block_fallback"


class SlotPhase(str, Enum):
    REUSABLE = "reusable"
    SUBMITTED = "submitted"
    READY = "ready"
    IN_USE = "in_use"


@dataclass(frozen=True)
class TensorMeta:
    name: str
    offset: int
    numel: int
    shape: tuple[int, ...]
    is_buffer: bool = False
    stride: tuple[int, ...] | None = None


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
    dtypes: tuple[DTypeManifest, ...]
    digest: str

    @property
    def pinned_bytes(self) -> int:
        return sum(d.pinned_bytes for d in self.dtypes)

    @property
    def chunk_count(self) -> int:
        return sum(len(d.chunks) for d in self.dtypes)


TensorSpec = tuple[str, torch.Tensor, bool]
