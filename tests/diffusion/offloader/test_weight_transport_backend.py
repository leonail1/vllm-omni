# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Section-30.4 consistency tests for the stage-2 weight transport backends.

For the same legal source contract every backend must deliver a bitwise
identical full output on every consumer rank.  The CPU reference is
``reconstruct_full_flat`` over ``pack_local_shard`` output; each backend's
per-chunk ``full_output`` is compared against it exactly (valid region plus
zero padding — the output buffer is pre-filled with NaN so any stale or
untouched byte fails the comparison).

Verified properties (design section 30.4):

* consistency: REFERENCE / GROUP_SCATTER_AG / PAIR_COPY /
  GROUP_PIPELINE_MEMCPY all reproduce the ``reconstruct_full_flat``
  reference bitwise, on every rank;
* generations: two consecutive generations leave no stale data behind;
* tail chunk: manifests deliberately end in a non-divisible tail chunk;
* observability: requested vs effective backend/layout are reported by
  ``select_transport`` and matched by ``create_transport_backend``;
* fallback: an unsupported request falls back to GROUP_SCATTER_AG on the
  *same* FS process group (no new group, proven by running the fallback
  selection end-to-end on one gloo group);
* bounded resources: counters stay exact, ``reset_generation`` is monotone,
  ``close`` releases backend-held resources and blocks further parts;
* no global synchronize: the platform ``synchronize`` is replaced with a
  stub that raises, and fake streams/events enforce that every
  ``wait_event`` targets an already-recorded event — correctness comes from
  stream/event ordering alone.

GROUP_PERSISTENT is skipped: it captures a ``torch.npu.NPUGraph`` around the
HCCL all-gather (design section 15.3) and has no gloo/CPU execution path.
"""

from __future__ import annotations

import datetime
from contextlib import contextmanager, nullcontext

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm_omni.diffusion.offloader.weight_transport_backend as wtb_module
from vllm_omni.diffusion.offloader.chunked_transport import (
    PartManifest,
    SourceLayout,
    TensorSpec,
    TransportBackendKind,
    build_part_manifest,
    pack_local_shard,
    reconstruct_full_flat,
)
from vllm_omni.diffusion.offloader.weight_transport_backend import (
    ChunkEvents,
    TransportCapability,
    TransportStreams,
    create_transport_backend,
    select_transport,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]

_WORLD_SIZE = 2
_CHUNK_SIZE_BYTES = 1024
_ALIGNMENT_BYTES = 256


# --------------------------------------------------------------------- #
#  Fake stream/event plumbing (no device, no global synchronize)        #
# --------------------------------------------------------------------- #


class _FakeStream:
    """Synchronous stand-in for a device stream.

    ``wait_event`` enforces the section-30.4 ordering contract: a stream may
    only wait on an event that has already been recorded.
    """

    def __init__(self, name: str = "stream") -> None:
        self.name = name

    def wait_event(self, event: _FakeEvent) -> None:
        if not event.recorded:
            raise AssertionError(f"{self.name} waits on event {event.name!r} that was never recorded")

    def wait_stream(self, _other: _FakeStream) -> None:
        return None

    def synchronize(self) -> None:
        # Per-stream join (used by backend close()); NOT a global synchronize.
        return None


class _FakeEvent:
    def __init__(self, name: str = "event") -> None:
        self.name = name
        self.recorded = False

    def record(self, _stream: _FakeStream) -> None:
        self.recorded = True


@contextmanager
def _fake_stream_ctx(_stream):
    yield None


def _forbidden_global_sync() -> None:
    raise AssertionError("weight transport must not use a global synchronize (design section 30.4)")


def _install_fake_platform(monkeypatch: pytest.MonkeyPatch | None = None) -> None:
    platform = wtb_module.current_omni_platform
    attrs = {
        "Stream": _FakeStream,
        "Event": _FakeEvent,
        "stream": _fake_stream_ctx,
        "synchronize": _forbidden_global_sync,
    }
    for name, value in attrs.items():
        if monkeypatch is not None:
            monkeypatch.setattr(platform, name, value)
        else:
            setattr(platform, name, value)


def _streams() -> TransportStreams:
    return TransportStreams(copy=_FakeStream("copy"), communication=_FakeStream("communication"))


def _trace(_name: str):
    return nullcontext()


def _recorded_event(name: str = "prior_last_use") -> _FakeEvent:
    event = _FakeEvent(name)
    event.record(_FakeStream("producer"))
    return event


# --------------------------------------------------------------------- #
#  Manifest / reference helpers                                          #
# --------------------------------------------------------------------- #


def _make_specs(generation: int) -> list[TensorSpec]:
    """Deterministic float32 specs; 600 elements => 2 full chunks + tail."""
    base = 1000.0 * generation
    w0 = (torch.arange(400, dtype=torch.float32) + base).reshape(100, 4)
    w1 = (torch.arange(120, dtype=torch.float32) + 0.5 + base).reshape(24, 5)
    buf = torch.arange(80, dtype=torch.float32) + 0.25 + base
    skipped = torch.arange(16, dtype=torch.int64)  # non-float: excluded from transport
    return [("attn.w0", w0, False), ("attn.w1", w1, False), ("attn.buf", buf, True), ("attn.skip", skipped, False)]


def _make_mixed_specs(generation: int) -> list[TensorSpec]:
    """Adds a float64 dtype group (200 elements => 1 full chunk + tail)."""
    base = 5000.0 * generation
    d0 = (torch.arange(128, dtype=torch.float64) + base).reshape(16, 8)
    d1 = torch.arange(72, dtype=torch.float64) + 0.5 + base
    return _make_specs(generation) + [("attn.d0", d0, False), ("attn.d1", d1, True)]


def _build_manifest(
    specs: list[TensorSpec],
    *,
    rank: int,
    world_size: int,
    source_layout: SourceLayout = SourceLayout.FS_SHARDED_HOST,
) -> PartManifest:
    return build_part_manifest(
        specs,
        block_id=0,
        part_id="attention",
        weight_shard_size=world_size,
        weight_shard_rank=rank,
        chunk_size_bytes=_CHUNK_SIZE_BYTES,
        alignment_bytes=_ALIGNMENT_BYTES,
        source_layout=source_layout,
    )


def _plain_allocator(numel: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(numel, dtype=dtype, device="cpu")


def _pack(specs: list[TensorSpec], manifest: PartManifest) -> dict[torch.dtype, torch.Tensor]:
    return pack_local_shard(specs, manifest, allocator=_plain_allocator)


def _reference_flat(specs: list[TensorSpec], world_size: int, dtype_manifest) -> torch.Tensor:
    """CPU reference reconstruction from every rank's FS-sharded Host shard."""
    shards = [
        _pack(specs, _build_manifest(specs, rank=r, world_size=world_size))[dtype_manifest.dtype]
        for r in range(world_size)
    ]
    return reconstruct_full_flat(shards, dtype_manifest)


def _assert_has_tail_chunk(manifest: PartManifest) -> None:
    assert any(chunk.valid_numel < chunk.padded_numel for dm in manifest.dtypes for chunk in dm.chunks), (
        "test must exercise a non-divisible tail chunk"
    )


# --------------------------------------------------------------------- #
#  Part submission driver                                                #
# --------------------------------------------------------------------- #


def _fs_source_fn(shard: dict[torch.dtype, torch.Tensor], *, with_local_input: bool):
    """Source contract for FS-sharded layouts (REFERENCE / GROUP_SCATTER_AG)."""

    def source_fn(dtype_manifest, chunk):
        local_shard = shard[dtype_manifest.dtype]
        source = local_shard[chunk.cpu_offset : chunk.cpu_offset + chunk.local_numel]
        local_input = (
            torch.full((chunk.local_numel,), float("nan"), dtype=dtype_manifest.dtype) if with_local_input else None
        )
        return source, local_input

    return source_fn


def _full_shard_source_fn(shard: dict[torch.dtype, torch.Tensor], *, is_owner: bool):
    """Source contract for full-Host layouts (PAIR_COPY leader / pipeline owner)."""

    def source_fn(dtype_manifest, chunk):
        if not is_owner:
            return None, None
        full_shard = shard[dtype_manifest.dtype]
        return full_shard[chunk.full_offset : chunk.full_offset + chunk.padded_numel], None

    return source_fn


def _submit_part(backend, manifest: PartManifest, *, source_fn, group, generation: int, prior_last_use):
    """Submit one full part; returns (per-chunk results, ready_event).

    Every ``full_output`` starts as NaN: the backend must overwrite the whole
    padded chunk (valid data + zero padding), so stale bytes cannot survive.
    """
    streams = _streams()
    backend.begin_part(streams, prior_last_use)
    completions = []
    results = []
    for dtype_manifest in manifest.dtypes:
        for chunk in dtype_manifest.chunks:
            source, local_input = source_fn(dtype_manifest, chunk)
            full_output = torch.full((chunk.padded_numel,), float("nan"), dtype=dtype_manifest.dtype)
            events = ChunkEvents(
                h2d_done=_FakeEvent("h2d_done"),
                transport_done=_FakeEvent("transport_done"),
                relay_done=_FakeEvent("relay_done"),
            )
            completions.append(
                backend.submit_chunk(
                    source=source,
                    local_input=local_input,
                    full_output=full_output,
                    chunk_meta=chunk,
                    streams=streams,
                    events=events,
                    group=group,
                    generation=generation,
                    non_blocking=False,
                    trace=_trace,
                )
            )
            results.append((dtype_manifest, chunk, full_output))
    ready = _FakeEvent("ready")
    returned = backend.finalize_part(completions, ready_event=ready, streams=streams)
    assert returned is ready
    assert ready.recorded
    return results, ready


def _check_part_results(results, reference_flat: dict[torch.dtype, torch.Tensor]) -> None:
    for dtype_manifest, chunk, full_output in results:
        expected = torch.zeros(chunk.padded_numel, dtype=dtype_manifest.dtype)
        flat = reference_flat[dtype_manifest.dtype]
        expected[: chunk.valid_numel] = flat[chunk.full_offset : chunk.full_offset + chunk.valid_numel]
        assert torch.equal(full_output, expected), (
            f"chunk {chunk.chunk_id} (dtype={dtype_manifest.dtype}) mismatch: "
            f"valid_numel={chunk.valid_numel} padded_numel={chunk.padded_numel}"
        )


def _capability(rank: int = 0, world_size: int = 1, **overrides) -> TransportCapability:
    kwargs = {
        "world_size": world_size,
        "rank": rank,
        "global_ranks": tuple(range(world_size)),
        "same_host": True,
        "p2p_supported": True,
    }
    kwargs.update(overrides)
    return TransportCapability(**kwargs)


# --------------------------------------------------------------------- #
#  Single-process tests                                                  #
# --------------------------------------------------------------------- #


def test_reference_single_rank_matches_cpu_reconstruct(monkeypatch: pytest.MonkeyPatch) -> None:
    """30.4 consistency + tail chunk for REFERENCE, multi-dtype manifest."""
    _install_fake_platform(monkeypatch)
    capability = _capability()
    selection = select_transport(TransportBackendKind.REFERENCE, SourceLayout.FS_SHARDED_HOST, capability)
    backend = create_transport_backend(selection, capability)
    assert backend.kind is TransportBackendKind.REFERENCE
    assert backend.requires_local_input is False

    specs = _make_mixed_specs(generation=0)
    manifest = _build_manifest(specs, rank=0, world_size=1)
    _assert_has_tail_chunk(manifest)
    shard = _pack(specs, manifest)
    reference = {dm.dtype: _reference_flat(specs, 1, dm) for dm in manifest.dtypes}

    results, _ready = _submit_part(
        backend,
        manifest,
        source_fn=_fs_source_fn(shard, with_local_input=False),
        group=None,
        generation=0,
        prior_last_use=None,
    )
    _check_part_results(results, reference)
    assert backend.counters.submitted_parts == 1
    assert backend.counters.submitted_chunks == manifest.chunk_count


def test_reference_consecutive_generations_leave_no_stale_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """30.4 generation isolation: generation 1 output carries no gen-0 bytes."""
    _install_fake_platform(monkeypatch)
    capability = _capability()
    selection = select_transport(TransportBackendKind.REFERENCE, SourceLayout.FS_SHARDED_HOST, capability)
    backend = create_transport_backend(selection, capability)

    ready = None
    for generation in (0, 1):
        backend.reset_generation(generation)
        specs = _make_specs(generation)
        manifest = _build_manifest(specs, rank=0, world_size=1)
        shard = _pack(specs, manifest)
        reference = {dm.dtype: _reference_flat(specs, 1, dm) for dm in manifest.dtypes}
        results, ready = _submit_part(
            backend,
            manifest,
            source_fn=_fs_source_fn(shard, with_local_input=False),
            group=None,
            generation=generation,
            prior_last_use=ready,
        )
        _check_part_results(results, reference)

    with pytest.raises(RuntimeError, match="moved backwards"):
        backend.reset_generation(0)
    assert backend.counters.submitted_parts == 2
    backend.close()


def test_requested_and_effective_backend_are_observable() -> None:
    """30.4 observability: requested/effective backend and layout are reported."""
    single = _capability()
    selection = select_transport(TransportBackendKind.REFERENCE, SourceLayout.FS_SHARDED_HOST, single)
    assert selection.requested_backend is TransportBackendKind.REFERENCE
    assert selection.effective_backend is TransportBackendKind.REFERENCE
    assert selection.fallback_reason is None
    assert create_transport_backend(selection, single).kind is TransportBackendKind.REFERENCE

    auto_single = select_transport(TransportBackendKind.AUTO, SourceLayout.FS_SHARDED_HOST, single)
    assert auto_single.requested_backend is TransportBackendKind.AUTO
    assert auto_single.effective_backend is TransportBackendKind.REFERENCE

    auto_fs = select_transport(TransportBackendKind.AUTO, SourceLayout.FS_SHARDED_HOST, _capability(world_size=2))
    assert auto_fs.effective_backend is TransportBackendKind.GROUP_SCATTER_AG

    # REFERENCE rejects a non-FS source layout instead of silently accepting it.
    bad = select_transport(TransportBackendKind.REFERENCE, SourceLayout.PAIR_LEADER_FULL_HOST, _capability())
    assert bad.effective_backend is TransportBackendKind.REFERENCE
    assert bad.effective_source_layout is SourceLayout.FS_SHARDED_HOST
    assert bad.fallback_reason is not None and "reference requires fs_sharded_host" in bad.fallback_reason


def test_unsupported_request_falls_back_without_changing_group() -> None:
    """30.4 fallback: unsupported requests degrade to the same FS group path."""
    # group_persistent without validated native support -> group_scatter_ag,
    # same fs_sharded_host layout, i.e. the same FS process group contract.
    selection = select_transport(
        TransportBackendKind.GROUP_PERSISTENT,
        SourceLayout.FS_SHARDED_HOST,
        _capability(world_size=2),  # native_persistent=False
    )
    assert selection.requested_backend is TransportBackendKind.GROUP_PERSISTENT
    assert selection.effective_backend is TransportBackendKind.GROUP_SCATTER_AG
    assert selection.requested_source_layout is SourceLayout.FS_SHARDED_HOST
    assert selection.effective_source_layout is SourceLayout.FS_SHARDED_HOST
    assert selection.fallback_reason is not None and "group_persistent" in selection.fallback_reason

    # pair_copy without validated same-host P2P edges -> same FS group fallback.
    no_p2p = select_transport(
        TransportBackendKind.PAIR_COPY,
        SourceLayout.PAIR_LEADER_FULL_HOST,
        _capability(world_size=2, same_host=False, p2p_supported=False),
    )
    assert no_p2p.requested_backend is TransportBackendKind.PAIR_COPY
    assert no_p2p.effective_backend is TransportBackendKind.GROUP_SCATTER_AG
    assert no_p2p.requested_source_layout is SourceLayout.PAIR_LEADER_FULL_HOST
    assert no_p2p.effective_source_layout is SourceLayout.FS_SHARDED_HOST
    assert no_p2p.fallback_reason is not None and "pair_copy" in no_p2p.fallback_reason


def test_reset_and_close_bound_backend_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    """30.4 bounded resources: exact counters, monotone generation, close()."""
    _install_fake_platform(monkeypatch)
    capability = _capability()
    selection = select_transport(TransportBackendKind.REFERENCE, SourceLayout.FS_SHARDED_HOST, capability)
    backend = create_transport_backend(selection, capability)

    specs = _make_specs(generation=0)
    manifest = _build_manifest(specs, rank=0, world_size=1)
    shard = _pack(specs, manifest)
    _submit_part(
        backend,
        manifest,
        source_fn=_fs_source_fn(shard, with_local_input=False),
        group=None,
        generation=0,
        prior_last_use=None,
    )
    assert backend.counters.submitted_parts == 1
    assert backend.counters.submitted_chunks == manifest.chunk_count
    assert backend.counters.backend_chunks == {TransportBackendKind.REFERENCE.value: manifest.chunk_count}

    backend.reset_counters()
    assert backend.counters.submitted_parts == 0
    assert backend.counters.submitted_chunks == 0
    assert backend.counters.backend_chunks == {}

    backend.reset_generation(0)
    backend.reset_generation(1)
    with pytest.raises(RuntimeError, match="moved backwards"):
        backend.reset_generation(0)

    backend.close()
    backend.close()  # close is idempotent
    with pytest.raises(RuntimeError, match="closed"):
        backend.begin_part(_streams(), None)


def test_pipeline_backend_close_releases_hop_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    """30.4 bounded resources: pipeline hop streams are released on close()."""
    _install_fake_platform(monkeypatch)
    capability = _capability(rank=0, world_size=2)
    selection = select_transport(
        TransportBackendKind.GROUP_PIPELINE_MEMCPY, SourceLayout.GROUP_OWNER_FULL_HOST, capability
    )
    assert selection.effective_backend is TransportBackendKind.GROUP_PIPELINE_MEMCPY
    backend = create_transport_backend(selection, capability)

    backend.begin_part(_streams(), _recorded_event())
    # Internal hop-stream pool is created lazily and must be bounded by close().
    assert backend._hop_streams is not None
    assert len(backend._hop_streams) == capability.world_size - 1

    backend.close()
    assert backend._hop_streams is None
    with pytest.raises(RuntimeError, match="closed"):
        backend.begin_part(_streams(), None)


# --------------------------------------------------------------------- #
#  Two-process gloo workers                                              #
# --------------------------------------------------------------------- #


def _init_gloo(rank: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=datetime.timedelta(seconds=120),
    )


def _run_two_generations(backend, group, source_layout: SourceLayout, rank: int, *, owner: bool) -> None:
    """Run generations 0 and 1 over *group* and check both against the reference."""
    ready = None
    chunk_total = 0
    for generation in (0, 1):
        backend.reset_generation(generation)
        specs = _make_specs(generation)
        manifest = _build_manifest(specs, rank=rank, world_size=_WORLD_SIZE, source_layout=source_layout)
        _assert_has_tail_chunk(manifest)
        shard = _pack(specs, manifest)
        reference = {dm.dtype: _reference_flat(specs, _WORLD_SIZE, dm) for dm in manifest.dtypes}
        if source_layout is SourceLayout.FS_SHARDED_HOST:
            source_fn = _fs_source_fn(shard, with_local_input=backend.requires_local_input)
        else:
            source_fn = _full_shard_source_fn(shard, is_owner=owner)
        results, ready = _submit_part(
            backend,
            manifest,
            source_fn=source_fn,
            group=group,
            generation=generation,
            prior_last_use=ready,
        )
        _check_part_results(results, reference)
        chunk_total += manifest.chunk_count

    try:
        backend.reset_generation(0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("reset_generation accepted a backwards generation")
    assert backend.counters.submitted_parts == 2
    assert backend.counters.submitted_chunks == chunk_total
    backend.close()


def _fs_consistency_worker(rank: int, init_file: str) -> None:
    """REFERENCE and GROUP_SCATTER_AG over one gloo FS group (30.4 core)."""
    _init_gloo(rank, init_file)
    try:
        _install_fake_platform()
        for kind in (TransportBackendKind.REFERENCE, TransportBackendKind.GROUP_SCATTER_AG):
            capability = _capability(rank=rank, world_size=_WORLD_SIZE)
            selection = select_transport(kind, SourceLayout.FS_SHARDED_HOST, capability)
            assert selection.effective_backend is kind
            assert selection.fallback_reason is None
            backend = create_transport_backend(selection, capability)
            assert backend.kind is kind
            assert backend.requires_local_input is True
            group = dist.new_group([0, 1])
            _run_two_generations(backend, group, SourceLayout.FS_SHARDED_HOST, rank, owner=False)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _pair_copy_worker(rank: int, init_file: str) -> None:
    """PAIR_COPY leader->follower copy over gloo P2P on the same FS group."""
    _init_gloo(rank, init_file)
    try:
        _install_fake_platform()
        fs_group = dist.new_group([0, 1])
        pair_group = dist.new_group([0, 1])
        capability = _capability(rank=rank, world_size=_WORLD_SIZE, pair_group=pair_group)
        selection = select_transport(TransportBackendKind.PAIR_COPY, SourceLayout.PAIR_LEADER_FULL_HOST, capability)
        assert selection.effective_backend is TransportBackendKind.PAIR_COPY
        assert selection.fallback_reason is None
        backend = create_transport_backend(selection, capability)
        assert backend.kind is TransportBackendKind.PAIR_COPY
        assert backend.requires_local_input is False
        _run_two_generations(backend, fs_group, SourceLayout.PAIR_LEADER_FULL_HOST, rank, owner=(rank % 2 == 0))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _pipeline_memcpy_worker(rank: int, init_file: str) -> None:
    """GROUP_PIPELINE_MEMCPY owner->rank1 relay over gloo broadcast hops."""
    _init_gloo(rank, init_file)
    try:
        _install_fake_platform()
        fs_group = dist.new_group([0, 1])
        hop_group = dist.new_group([0, 1])
        capability = _capability(rank=rank, world_size=_WORLD_SIZE, pipeline_hop_groups=(hop_group,))
        selection = select_transport(
            TransportBackendKind.GROUP_PIPELINE_MEMCPY, SourceLayout.GROUP_OWNER_FULL_HOST, capability
        )
        assert selection.effective_backend is TransportBackendKind.GROUP_PIPELINE_MEMCPY
        assert selection.fallback_reason is None
        backend = create_transport_backend(selection, capability)
        assert backend.kind is TransportBackendKind.GROUP_PIPELINE_MEMCPY
        assert backend.requires_local_input is False
        _run_two_generations(backend, fs_group, SourceLayout.GROUP_OWNER_FULL_HOST, rank, owner=(rank == 0))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _fallback_same_group_worker(rank: int, init_file: str) -> None:
    """A group_persistent request falls back to group_scatter_ag and then runs
    correctly on the SAME, unchanged gloo FS group (30.4 fallback)."""
    _init_gloo(rank, init_file)
    try:
        _install_fake_platform()
        fs_group = dist.new_group([0, 1])
        capability = _capability(rank=rank, world_size=_WORLD_SIZE)  # native_persistent=False
        selection = select_transport(TransportBackendKind.GROUP_PERSISTENT, SourceLayout.FS_SHARDED_HOST, capability)
        assert selection.requested_backend is TransportBackendKind.GROUP_PERSISTENT
        assert selection.effective_backend is TransportBackendKind.GROUP_SCATTER_AG
        assert selection.effective_source_layout is SourceLayout.FS_SHARDED_HOST
        assert selection.fallback_reason is not None
        backend = create_transport_backend(selection, capability)
        assert backend.kind is TransportBackendKind.GROUP_SCATTER_AG
        _run_two_generations(backend, fs_group, SourceLayout.FS_SHARDED_HOST, rank, owner=False)
        dist.barrier()
    finally:
        dist.destroy_process_group()


# --------------------------------------------------------------------- #
#  Two-process gloo test entry points                                    #
# --------------------------------------------------------------------- #

_GLOO_UNAVAILABLE = not dist.is_available() or not dist.is_gloo_available()


@pytest.mark.skipif(_GLOO_UNAVAILABLE, reason="gloo is required")
def test_fs_backends_two_rank_consistency(tmp_path) -> None:
    mp.spawn(_fs_consistency_worker, args=(str(tmp_path / "gloo_fs"),), nprocs=_WORLD_SIZE, join=True)


@pytest.mark.skipif(_GLOO_UNAVAILABLE, reason="gloo is required")
def test_pair_copy_two_rank_consistency(tmp_path) -> None:
    mp.spawn(_pair_copy_worker, args=(str(tmp_path / "gloo_pair"),), nprocs=_WORLD_SIZE, join=True)


@pytest.mark.skipif(_GLOO_UNAVAILABLE, reason="gloo is required")
def test_pipeline_memcpy_two_rank_consistency(tmp_path) -> None:
    mp.spawn(_pipeline_memcpy_worker, args=(str(tmp_path / "gloo_pipe"),), nprocs=_WORLD_SIZE, join=True)


@pytest.mark.skipif(_GLOO_UNAVAILABLE, reason="gloo is required")
def test_fallback_keeps_same_fs_group(tmp_path) -> None:
    mp.spawn(_fallback_same_group_worker, args=(str(tmp_path / "gloo_fallback"),), nprocs=_WORLD_SIZE, join=True)


@pytest.mark.skip(
    reason=(
        "group_persistent captures a torch.npu.NPUGraph around the HCCL all-gather "
        "(design section 15.3) and requires native_persistent validation plus NPU "
        "devices; it has no gloo/CPU execution path, so section-30.4 covers it via "
        "the selection/fallback tests instead"
    )
)
def test_group_persistent_two_rank_consistency() -> None:
    pass
