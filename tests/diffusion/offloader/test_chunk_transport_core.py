# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from vllm_omni.diffusion.offloader._chunk_partition import build_part_manifest
from vllm_omni.diffusion.offloader._chunk_slots import ChunkTransportState, ChunkedWeightTransport
from vllm_omni.diffusion.offloader._chunk_types import ChunkMeta, SlotPhase, TransportBackendKind
from vllm_omni.diffusion.offloader.chunked_transport import WeightLayout, pack_local_shard
from vllm_omni.diffusion.offloader.weight_transport_backend import (
    ChunkEvents,
    ReferenceBackend,
    TransportCapability,
    TransportStreams,
    create_transport_backend,
    select_transport,
)


def _cpu_alloc(numel: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(numel, dtype=dtype, device="cpu")


def _specs():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(128, 64, generator=g, dtype=torch.bfloat16)
    b = torch.randn(64, generator=g, dtype=torch.bfloat16)
    skip = torch.randn(32, 32, generator=g, dtype=torch.float32)
    return [("w", w, False), ("b", b, False), ("skip", skip, True)]


@pytest.mark.parametrize("shard_size,rank,layout_name", [
    (1, 0, "CHUNK_MAJOR"),
    (2, 0, "CHUNK_MAJOR"),
    (2, 1, "CHUNK_MAJOR"),
    (4, 2, "CHUNK_MAJOR"),
    (2, 0, "WHOLE_BLOCK"),
])
def test_manifest_digest_stable(shard_size, rank, layout_name):
    specs = _specs()
    first = build_part_manifest(
        specs, block_id=7, part_id="block", weight_shard_size=shard_size, weight_shard_rank=rank,
        chunk_size_bytes=8 * 1024, alignment_bytes=256, layout=getattr(WeightLayout, layout_name),
    )
    second = build_part_manifest(
        specs, block_id=7, part_id="block", weight_shard_size=shard_size, weight_shard_rank=rank,
        chunk_size_bytes=8 * 1024, alignment_bytes=256, layout=getattr(WeightLayout, layout_name),
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


def test_slot_lifecycle_and_idempotent_begin():
    st = ChunkTransportState(block_id=3, slot_count=2)
    t0 = st.begin(output_slot=0, chunk_count=4, request_generation=1, part_id="block", last_collective_key=(1, 3, 0, "ab"))
    assert st.slots[0].phase is SlotPhase.SUBMITTED
    again = st.begin(output_slot=0, chunk_count=4, request_generation=1, part_id="block", last_collective_key=(1, 3, 0, "ab"))
    assert again is t0
    with pytest.raises(RuntimeError):
        st.begin(output_slot=0, chunk_count=5, request_generation=1, part_id="block")
    st.mark_ready(t0)
    st.mark_in_use(t0)
    st.release(t0, last_use_event="evt")
    assert st.slots[0].phase is SlotPhase.REUSABLE
    t1 = st.begin(output_slot=1, chunk_count=4)
    assert t1.output_slot == 1


def test_transport_state_alias():
    tr = ChunkedWeightTransport(9)
    assert tr.state is tr
    assert tr.block_id == 9


def test_prepare_rejects_size_mismatch():
    w = torch.zeros(32, 8)
    manifest = build_part_manifest(
        [("w", w, False)], block_id=1, part_id="block", weight_shard_size=1, weight_shard_rank=0,
        chunk_size_bytes=64 * 1024 * 1024,
    )
    tr = ChunkedWeightTransport(1)
    with pytest.raises(ValueError):
        tr.prepare(manifest, {w.dtype: torch.zeros(3)})


class FakeEvent:
    def __init__(self, name: str):
        self.name = name
        self.records: list[object] = []

    def record(self, stream):
        self.records.append(stream)


class FakeStream:
    def __init__(self, name: str):
        self.name = name
        self.waits: list[object] = []

    def wait_event(self, event):
        self.waits.append(getattr(event, "name", event))


@contextmanager
def _trace(kind: str):
    yield


def test_fs_submit_event_order():
    copy, comm = FakeStream("copy"), FakeStream("comm")
    h2d, done, reusable = FakeEvent("h2d"), FakeEvent("done"), FakeEvent("reusable")
    source = torch.arange(8, dtype=torch.float32)
    local = torch.zeros(8)
    full = torch.zeros(16)
    calls: list[str] = []

    def collective():
        calls.append("ag")
        full[:8].copy_(local)
        full[8:].copy_(local)

    backend = ReferenceBackend(TransportCapability(world_size=2, rank=0, global_ranks=(0, 1)))
    completion = backend._submit_fs_chunk(
        source, local, full,
        ChunkMeta(0, 0, 0, 8, 8, 8),
        TransportStreams(copy, comm),
        ChunkEvents(h2d, done, reusable),
        non_blocking=False, trace=_trace, collective=collective,
    )
    assert copy.waits == ["reusable"]
    assert comm.waits == ["h2d"]
    assert h2d.records == [copy]
    assert done.records == [comm]
    assert calls == ["ag"]
    assert completion.event is done
    torch.testing.assert_close(local, source)


def test_select_auto_multi_rank():
    cap = TransportCapability(4, 1, (0, 1, 2, 3))
    sel = select_transport(TransportBackendKind.AUTO, cap)
    assert sel.effective_backend is TransportBackendKind.GROUP_SCATTER_AG
    backend = create_transport_backend(sel, cap)
    assert backend.kind is TransportBackendKind.GROUP_SCATTER_AG


def test_select_auto_single_rank():
    cap = TransportCapability(1, 0, (0,))
    sel = select_transport(TransportBackendKind.AUTO, cap)
    assert sel.effective_backend is TransportBackendKind.REFERENCE
    backend = create_transport_backend(sel, cap)
    assert backend.kind is TransportBackendKind.REFERENCE
    assert backend.requires_local_input is False
