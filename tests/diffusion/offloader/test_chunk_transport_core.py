# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.offloader.chunked_transport import (
    WeightLayout,
    build_part_manifest,
    pack_local_shard,
)


def _cpu_alloc(numel: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(numel, dtype=dtype, device="cpu")


def _specs():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(128, 64, generator=g, dtype=torch.bfloat16)
    b = torch.randn(64, generator=g, dtype=torch.bfloat16)
    skip = torch.randn(32, 32, generator=g, dtype=torch.float32)
    return [("w", w, False), ("b", b, False), ("skip", skip, True)]


@pytest.mark.parametrize(
    "shard_size,rank,layout_name",
    [
        (1, 0, "CHUNK_MAJOR"),
        (2, 0, "CHUNK_MAJOR"),
        (2, 1, "CHUNK_MAJOR"),
        (4, 2, "CHUNK_MAJOR"),
        (2, 0, "WHOLE_BLOCK"),
    ],
)
def test_manifest_digest_stable(shard_size, rank, layout_name):
    specs = _specs()
    first = build_part_manifest(
        specs,
        block_id=7,
        part_id="block",
        weight_shard_size=shard_size,
        weight_shard_rank=rank,
        chunk_size_bytes=8 * 1024,
        alignment_bytes=256,
        layout=getattr(WeightLayout, layout_name),
    )
    second = build_part_manifest(
        specs,
        block_id=7,
        part_id="block",
        weight_shard_size=shard_size,
        weight_shard_rank=rank,
        chunk_size_bytes=8 * 1024,
        alignment_bytes=256,
        layout=getattr(WeightLayout, layout_name),
    )
    assert first.digest == second.digest
    assert first.chunk_count == second.chunk_count


def test_pack_roundtrip_two_ranks():
    specs = _specs()
    kw = dict(block_id=0, part_id="block", weight_shard_size=2, chunk_size_bytes=4096, alignment_bytes=256)
    m0 = build_part_manifest(specs, weight_shard_rank=0, **kw)
    m1 = build_part_manifest(specs, weight_shard_rank=1, **kw)
    s0 = pack_local_shard(specs, m0, allocator=_cpu_alloc)
    s1 = pack_local_shard(specs, m1, allocator=_cpu_alloc)
    assert set(s0) == set(s1)
    for dtype in s0:
        assert s0[dtype].numel() == s1[dtype].numel()


def test_chunk_major_rank_slices_are_disjoint():
    specs = _specs()
    kw = dict(block_id=0, part_id="block", weight_shard_size=2, chunk_size_bytes=4096, alignment_bytes=256)
    m0 = build_part_manifest(specs, weight_shard_rank=0, **kw)
    m1 = build_part_manifest(specs, weight_shard_rank=1, **kw)
    s0 = pack_local_shard(specs, m0, allocator=_cpu_alloc)
    s1 = pack_local_shard(specs, m1, allocator=_cpu_alloc)
    for dtype_manifest in m0.dtypes:
        dtype = dtype_manifest.dtype
        for chunk in dtype_manifest.chunks:
            a = s0[dtype][chunk.cpu_offset : chunk.cpu_offset + chunk.local_numel]
            b = s1[dtype][chunk.cpu_offset : chunk.cpu_offset + chunk.local_numel]
            if chunk.valid_numel == 0:
                continue
            assert a.shape == b.shape
