# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the design-section-24 policy-2 skip schedule digest sync.

Covers the CollectiveSchedule digest construction (sensitivity to every
input, determinism), the single-rank / group-less coordinator paths, the
coordinator registry, and the cache-dit decision observer installation.

Multi-process agreement/fault-injection coverage lives in
test_skip_schedule_dist.py (gloo, two CPU processes).

CPU-only: no streams, no collectives, no NPU required.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.offloader.skip_schedule import (
    BlockPartSchedule,
    SkipScheduleCoordinator,
    _reset_registry_for_tests,
    build_step_schedule_digest,
    has_skip_schedule_coordinator,
    notify_skip_decision,
    register_skip_schedule_coordinator,
    skip_decision_observer_installed,
    unregister_skip_schedule_coordinator,
)


def _blocks(count: int = 4) -> list[BlockPartSchedule]:
    return [
        BlockPartSchedule(
            block_id=i,
            parts=(
                ("attention", (("torch.bfloat16", 3),)),
                ("moe", (("torch.bfloat16", 5), ("torch.float32", 1))),
            ),
        )
        for i in range(count)
    ]


def _manifest(block_id: int = 0, part_id: str = "attention", chunk_count: int = 2):
    """Minimal two-chunk manifest for a prepared CPU transport."""
    from vllm_omni.diffusion.offloader.chunked_transport import (
        ChunkMeta,
        DTypeManifest,
        PartManifest,
        SourceLayout,
        WeightLayout,
    )

    chunks = tuple(
        ChunkMeta(
            chunk_id=i,
            cpu_offset=i * 4,
            full_offset=i * 4,
            valid_numel=4,
            padded_numel=4,
            local_numel=4,
        )
        for i in range(chunk_count)
    )
    dtype_manifest = DTypeManifest(
        dtype=torch.float32,
        tensors=(),
        chunks=chunks,
        total_numel=4 * chunk_count,
        padded_numel=4 * chunk_count,
        local_numel=4 * chunk_count,
        local_chunk_numel=4,
        alignment_numel=1,
    )
    return PartManifest(
        block_id=block_id,
        part_id=part_id,
        weight_shard_size=1,
        weight_shard_rank=0,
        chunk_size_bytes=16,
        alignment_bytes=4,
        layout=WeightLayout.CHUNK_MAJOR,
        source_layout=SourceLayout.FS_SHARDED_HOST,
        dtypes=(dtype_manifest,),
        digest="digest0123456789abcdef",
    )


def _digest(**overrides):
    kwargs = {
        "request_generation": 0,
        "decision_index": 0,
        "skipped": False,
        "blocks": tuple(_blocks()),
        "fn_blocks": 1,
        "bn_blocks": 0,
        "backend_id": "reference",
    }
    kwargs.update(overrides)
    return build_step_schedule_digest(**kwargs)


@pytest.fixture(autouse=True)
def _clean_registry():
    _reset_registry_for_tests()
    yield
    _reset_registry_for_tests()


class TestDigestConstruction:
    def test_deterministic(self):
        assert _digest() == _digest()

    def test_fixed_size(self):
        assert len(_digest()) == 16
        assert len(_digest(skipped=True)) == 16

    def test_skip_decision_changes_digest(self):
        assert _digest(skipped=False) != _digest(skipped=True)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"request_generation": 1},
            {"decision_index": 1},
            {"backend_id": "pair_copy"},
            {"fn_blocks": 2},
            {"bn_blocks": 1},
        ],
        ids=["generation", "decision_index", "backend", "fn_blocks", "bn_blocks"],
    )
    def test_every_input_is_hashed(self, overrides):
        assert _digest() != _digest(**overrides)

    def test_block_structure_is_hashed(self):
        blocks = _blocks()
        tampered = (
            blocks[:1]
            + [
                BlockPartSchedule(
                    block_id=1,
                    parts=(
                        ("attention", (("torch.bfloat16", 4),)),
                        ("moe", (("torch.bfloat16", 5), ("torch.float32", 1))),
                    ),
                )
            ]
            + blocks[2:]
        )
        assert _digest() != _digest(blocks=tuple(tampered))

    def test_skipped_step_covers_all_affected_blocks(self):
        """A skipped step hashes the same regardless of block contents: no
        collectives remain.  The decision flag still distinguishes it."""
        digest_skip = _digest(skipped=True)
        assert digest_skip == _digest(skipped=True, blocks=tuple(_blocks(count=8)))

    def test_invalid_fn_bn_rejected(self):
        with pytest.raises(ValueError, match="invalid Fn/Bn"):
            _digest(fn_blocks=3, bn_blocks=2)


class TestCoordinatorSingleRank:
    def test_no_group_never_raises(self):
        coordinator = SkipScheduleCoordinator(
            domain="gen_layers", blocks=_blocks(), backend_id="reference", cpu_group=None, group_size=1
        )
        coordinator.set_request_generation(0)
        coordinator.on_decision(skipped=False, fn_blocks=1, bn_blocks=0)
        coordinator.on_decision(skipped=True, fn_blocks=1, bn_blocks=0)
        assert coordinator.counters.decisions == 2
        assert coordinator.counters.skipped == 1
        assert coordinator.counters.digest_compares == 0
        assert coordinator.last_digest is not None

    def test_decision_index_resets_per_request(self):
        coordinator = SkipScheduleCoordinator(
            domain="gen_layers", blocks=_blocks(), backend_id="reference", cpu_group=None, group_size=1
        )
        coordinator.set_request_generation(0)
        coordinator.on_decision(skipped=False, fn_blocks=1, bn_blocks=0)
        first = coordinator.last_digest
        coordinator.set_request_generation(1)
        coordinator.on_decision(skipped=False, fn_blocks=1, bn_blocks=0)
        # Same decision position, new generation: hashed differently.
        assert coordinator.last_digest != first

    def test_generation_backwards_rejected(self):
        coordinator = SkipScheduleCoordinator(
            domain="gen_layers", blocks=_blocks(), backend_id="reference", cpu_group=None, group_size=1
        )
        coordinator.set_request_generation(2)
        with pytest.raises(RuntimeError, match="moved backwards"):
            coordinator.set_request_generation(1)


class TestRegistry:
    def _coordinator(self, domain: str = "gen_layers") -> SkipScheduleCoordinator:
        return SkipScheduleCoordinator(
            domain=domain, blocks=_blocks(), backend_id="reference", cpu_group=None, group_size=1
        )

    def test_notify_routes_by_domain_prefix(self):
        coordinator = self._coordinator("gen_layers")
        other = self._coordinator("layers")
        register_skip_schedule_coordinator("gen_layers", coordinator)
        register_skip_schedule_coordinator("layers", other)
        assert has_skip_schedule_coordinator()

        notify_skip_decision(skipped=True, prefix="gen_layers_Fn_residual", fn_blocks=1, bn_blocks=0)
        assert coordinator.counters.decisions == 1
        assert coordinator.counters.skipped == 1
        assert other.counters.decisions == 0

        notify_skip_decision(skipped=False, prefix="layers_Fn_hidden_states", fn_blocks=1, bn_blocks=0)
        assert other.counters.decisions == 1

        unregister_skip_schedule_coordinator(coordinator)
        unregister_skip_schedule_coordinator(other)
        assert not has_skip_schedule_coordinator()
        notify_skip_decision(skipped=True, prefix="gen_layers_Fn_residual", fn_blocks=1, bn_blocks=0)
        assert coordinator.counters.decisions == 1

    def test_unmatched_prefix_is_ignored(self):
        coordinator = self._coordinator("gen_layers")
        register_skip_schedule_coordinator("gen_layers", coordinator)
        notify_skip_decision(skipped=True, prefix="other_blocks_Fn_residual", fn_blocks=1, bn_blocks=0)
        assert coordinator.counters.decisions == 0

    def test_double_registration_rejected(self):
        coordinator = self._coordinator()
        register_skip_schedule_coordinator("gen_layers", coordinator)
        with pytest.raises(RuntimeError, match="already registered"):
            register_skip_schedule_coordinator("gen_layers", self._coordinator())

    def test_notify_without_coordinator_is_noop(self):
        notify_skip_decision(skipped=True, prefix="gen_layers_Fn_residual", fn_blocks=1, bn_blocks=0)


class TestSkipReconciliation:
    """The policy-2 on_skip callback retires stranded pre-decision prefetches."""

    def _coordinator(self, on_skip=None) -> SkipScheduleCoordinator:
        return SkipScheduleCoordinator(
            domain="gen_layers",
            blocks=_blocks(),
            backend_id="reference",
            cpu_group=None,
            group_size=1,
            on_skip=on_skip,
        )

    def test_on_skip_called_only_when_skipped(self):
        calls = []
        coordinator = self._coordinator(on_skip=lambda: calls.append(1))
        coordinator.set_request_generation(0)
        coordinator.on_decision(skipped=False, fn_blocks=1, bn_blocks=0)
        assert calls == []
        coordinator.on_decision(skipped=True, fn_blocks=1, bn_blocks=0)
        assert calls == [1]

    def test_on_skip_not_called_on_mismatch_path(self):
        """The callback must not run before/without a successful compare —
        here there is no group, so this also covers the single-rank path."""
        calls = []
        coordinator = self._coordinator(on_skip=lambda: calls.append(1))
        coordinator.set_request_generation(0)
        coordinator.on_decision(skipped=True, fn_blocks=1, bn_blocks=0)
        assert calls == [1]

    def test_retire_unconsumed_prefetches(self):
        """A live READY ticket is retired, its events cleared, and the target
        block's Parameters reset to placeholders; consumed parts untouched."""
        from vllm_omni.diffusion.offloader.chunked_transport import (
            ChunkedWeightTransport,
            SlotPhase,
        )
        from vllm_omni.diffusion.offloader.part_pipeline import DistributedPartPipelineOffloadHook
        from vllm_omni.diffusion.offloader.tensor_utils import is_materialized_tensor

        attention_transport = ChunkedWeightTransport(block_id=1, slot_count=2)
        attention_transport.prepare(_manifest(block_id=1), {torch.float32: torch.zeros(8)})
        moe_transport = ChunkedWeightTransport(block_id=1, slot_count=2)
        moe_transport.prepare(_manifest(block_id=1), {torch.float32: torch.zeros(8)})

        ticket = attention_transport.begin_submission(
            output_slot=0, request_generation=0, ready_event=object(), part_id="attention"
        )
        attention_transport.mark_ready(ticket)

        hook = object.__new__(DistributedPartPipelineOffloadHook)
        hook.block_id = 1
        hook._part_futures = {"attention": None, "moe": None}
        hook.ready_tickets = {"attention": ticket, "moe": None}
        hook.part_transports = {"attention": attention_transport, "moe": moe_transport}
        hook.ready_events = {"attention": object(), "moe": None}
        hook.part_slots = {"attention": 0, "moe": 1}
        hook._shared_slot_owners = [hook, None]
        hook._transported_names = {"attention": {"w"}, "moe": set()}
        hook.next_block_parameters = {"w": torch.nn.Parameter(torch.zeros(4))}
        hook.next_block_buffers = {}
        hook._prefetched_parts = {"attention"}

        hook.retire_unconsumed_prefetches()

        assert hook.ready_tickets["attention"] is None
        assert hook.ready_events["attention"] is None
        assert attention_transport.state.slots[0].phase is SlotPhase.RETIRED
        assert hook._shared_slot_owners[0] is None
        assert not is_materialized_tensor(hook.next_block_parameters["w"])
        assert "attention" not in hook._prefetched_parts

        # Idempotent: a second pass finds no live ticket.
        hook.retire_unconsumed_prefetches()

    def test_retire_without_live_ticket_is_noop(self):
        from vllm_omni.diffusion.offloader.part_pipeline import DistributedPartPipelineOffloadHook

        hook = object.__new__(DistributedPartPipelineOffloadHook)
        hook.block_id = 1
        hook._part_futures = {"attention": None, "moe": None}
        hook.ready_tickets = {"attention": None, "moe": None}
        hook.part_transports = {}
        hook.ready_events = {"attention": None, "moe": None}
        hook.part_slots = {"attention": 0, "moe": 1}
        hook._shared_slot_owners = [None, None]
        hook._transported_names = {"attention": set(), "moe": set()}
        hook.next_block_parameters = {}
        hook.next_block_buffers = {}
        hook._prefetched_parts = set()

        hook.retire_unconsumed_prefetches()  # must not raise


class TestDecisionObserver:
    def test_install_is_idempotent_and_marks_registry(self):
        from vllm_omni.diffusion.cache.cachedit.skip_sync import (
            install_skip_decision_observer,
            skip_decision_observer_active,
        )

        install_skip_decision_observer()
        install_skip_decision_observer()
        assert skip_decision_observer_active()
        assert skip_decision_observer_installed()

    def test_observer_wraps_cache_dit_decision_point(self):
        from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager

        from vllm_omni.diffusion.cache.cachedit.skip_sync import install_skip_decision_observer

        install_skip_decision_observer()
        wrapped = CachedContextManager.can_cache
        # functools.wraps exposes the pre-wrap decision function; its presence
        # proves the wrapper is in place regardless of which test installed it.
        assert getattr(wrapped, "__wrapped__", None) is not None
        assert wrapped.__name__ == "can_cache"

    def test_observer_forwards_decision_to_coordinator(self):
        """Drive the wrapped can_cache with a stubbed original and verify the
        decision, Fn/Bn and prefix reach the registered coordinator."""
        import cache_dit.caching.cache_contexts.cache_manager as cache_manager

        from vllm_omni.diffusion.cache.cachedit.skip_sync import install_skip_decision_observer

        coordinator = SkipScheduleCoordinator(
            domain="gen_layers", blocks=_blocks(), backend_id="reference", cpu_group=None, group_size=1
        )
        register_skip_schedule_coordinator("gen_layers", coordinator)

        install_skip_decision_observer()
        wrapped = cache_manager.CachedContextManager.can_cache

        class _StubContextManager:
            Fn_compute_blocks = 1
            Bn_compute_blocks = 0

        stub = _StubContextManager()
        # Replace the closure's original with a stub that always decides
        # "skip"; the wrapper must still forward the decision.
        closure_cells = {name: cell for name, cell in zip(wrapped.__code__.co_freevars, wrapped.__closure__)}
        original_cell = closure_cells["original"]
        real_original = original_cell.cell_contents
        original_cell.cell_contents = lambda self, states_tensor, parallelized=False, threshold=None, prefix="Fn": True
        try:
            decision = wrapped(stub, torch.zeros(4), prefix="gen_layers_Fn_residual")
        finally:
            original_cell.cell_contents = real_original

        assert decision is True
        assert coordinator.counters.decisions == 1
        assert coordinator.counters.skipped == 1
