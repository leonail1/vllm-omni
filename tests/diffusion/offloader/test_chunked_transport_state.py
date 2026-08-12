# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the design-section-25 slot state machine.

Covers the full lifecycle

    EMPTY -> PREFETCH_ENQUEUED -> READY -> BOUND -> IN_USE -> RETIRED -> REUSABLE

plus the two failure exits (CANCELLED / POISONED_PROCESS_GROUP), the
section-25 transition checklist (generation, block/part, backend, output
slot, prior last-use, expected collective key) and the section-24 key
conflict rules (same-key re-prefetch is idempotent, a different key on a
live slot fails fast).

CPU-only: no streams, no collectives, no NPU required.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.offloader.chunked_transport import (
    ChunkedWeightTransport,
    ChunkMeta,
    ChunkTransportState,
    DTypeManifest,
    PartManifest,
    SlotPhase,
    SourceLayout,
    TransferTicket,
    WeightLayout,
)

_KEY = (0, 0, "attention", 0, "digest0123456789ab")


def _begin(state: ChunkTransportState, slot: int = 0, **overrides) -> TransferTicket:
    kwargs = {
        "output_slot": slot,
        "chunk_count": 2,
        "request_generation": 0,
        "part_id": "attention",
        "ready_event": object(),
        "backend_id": "reference",
        "last_collective_key": _KEY,
    }
    kwargs.update(overrides)
    return state.begin(**kwargs)


def _drive_to_in_use(state: ChunkTransportState, slot: int = 0, event: object | None = None) -> TransferTicket:
    ticket = _begin(state, slot)
    state.mark_ready(ticket)
    state.mark_bound(ticket)
    state.mark_in_use(ticket)
    return ticket


class TestMainPath:
    def test_full_lifecycle_phases(self):
        state = ChunkTransportState(block_id=0)
        assert state.slots[0].phase is SlotPhase.EMPTY

        ticket = _begin(state)
        assert state.slots[0].phase is SlotPhase.PREFETCH_ENQUEUED
        assert state.is_current(ticket)

        state.mark_ready(ticket)
        assert state.slots[0].phase is SlotPhase.READY

        state.mark_bound(ticket)
        assert state.slots[0].phase is SlotPhase.BOUND

        state.mark_in_use(ticket)
        assert state.slots[0].phase is SlotPhase.IN_USE

        state.record_last_use(ticket, object())
        assert state.slots[0].phase is SlotPhase.RETIRED
        # Retired: no longer current, ticket kept for audit.
        assert not state.is_current(ticket)

        state.confirm_reusable(ticket)
        assert state.slots[0].phase is SlotPhase.REUSABLE

        # Slot takes a fresh transfer with a new forward generation.
        next_ticket = _begin(state, request_generation=1)
        assert state.slots[0].phase is SlotPhase.PREFETCH_ENQUEUED
        assert next_ticket.forward_generation == ticket.forward_generation + 1

    def test_begin_on_retired_slot_auto_reuses(self):
        state = ChunkTransportState(block_id=0)
        ticket = _drive_to_in_use(state)
        state.record_last_use(ticket, object())
        assert state.slots[0].phase is SlotPhase.RETIRED

        next_ticket = _begin(state, request_generation=1)
        assert next_ticket is not ticket
        assert next_ticket.forward_generation == ticket.forward_generation + 1
        assert state.slots[0].phase is SlotPhase.PREFETCH_ENQUEUED

    def test_reset_after_retire_returns_to_empty(self):
        state = ChunkTransportState(block_id=0)
        ticket = _drive_to_in_use(state)
        state.record_last_use(ticket, object())
        state.reset()
        assert all(slot.phase is SlotPhase.EMPTY for slot in state.slots)

        fresh = _begin(state)
        assert fresh.forward_generation == 1


class TestKeyConflictRules:
    def test_same_key_resubmission_returns_existing_ticket(self):
        state = ChunkTransportState(block_id=0)
        ticket = _begin(state)
        again = _begin(state)
        assert again is ticket
        assert state.slots[0].phase is SlotPhase.PREFETCH_ENQUEUED

    @pytest.mark.parametrize(
        "overrides",
        [
            {"request_generation": 1},
            {"part_id": "moe"},
            {"backend_id": "pair_copy"},
            {"last_collective_key": (0, 0, "attention", 0, "ffffffffffffffff")},
            {"chunk_count": 3},
        ],
        ids=["generation", "part", "backend", "collective_key", "chunk_count"],
    )
    def test_different_key_on_live_slot_fails(self, overrides):
        state = ChunkTransportState(block_id=0)
        _begin(state)
        with pytest.raises(RuntimeError, match="different collective key"):
            _begin(state, **overrides)

    def test_different_key_allowed_after_retire(self):
        state = ChunkTransportState(block_id=0)
        ticket = _drive_to_in_use(state)
        state.record_last_use(ticket, object())
        next_ticket = _begin(state, request_generation=1, part_id="moe")
        assert next_ticket.part_id == "moe"


class TestIllegalTransitions:
    def test_mark_ready_requires_enqueued(self):
        state = ChunkTransportState(block_id=0)
        ticket = _begin(state)
        state.mark_ready(ticket)
        with pytest.raises(RuntimeError, match="cannot mark ready slot ready"):
            state.mark_ready(ticket)

    def test_mark_bound_requires_ready(self):
        state = ChunkTransportState(block_id=0)
        ticket = _begin(state)
        with pytest.raises(RuntimeError, match="cannot bind prefetch_enqueued slot"):
            state.mark_bound(ticket)

    def test_attach_without_bind_rejected(self):
        """READY -> IN_USE directly would read storage Parameters do not view."""
        state = ChunkTransportState(block_id=0)
        ticket = _begin(state)
        state.mark_ready(ticket)
        with pytest.raises(RuntimeError, match="cannot attach consumer to ready slot"):
            state.mark_in_use(ticket)

    def test_record_last_use_requires_published_slot(self):
        state = ChunkTransportState(block_id=0)
        ticket = _begin(state)
        with pytest.raises(RuntimeError, match="cannot retire prefetch_enqueued slot"):
            state.record_last_use(ticket, object())

    def test_confirm_reusable_requires_retired(self):
        state = ChunkTransportState(block_id=0)
        ticket = _begin(state)
        state.mark_ready(ticket)
        with pytest.raises(RuntimeError, match="cannot confirm reusable on ready slot"):
            state.confirm_reusable(ticket)

    def test_stale_ticket_rejected_after_retire_and_reuse(self):
        state = ChunkTransportState(block_id=0)
        old = _drive_to_in_use(state)
        state.record_last_use(old, object())
        _begin(state, request_generation=1)
        with pytest.raises(RuntimeError, match="stale transfer ticket"):
            state.mark_ready(old)

    def test_stale_ticket_rejected_after_retire(self):
        state = ChunkTransportState(block_id=0)
        old = _drive_to_in_use(state)
        state.record_last_use(old, object())
        # The slot still holds the retired ticket, but it is no longer
        # current: transitions on it must fail.
        with pytest.raises(RuntimeError, match="cannot mark retired slot ready"):
            state.mark_ready(old)


class TestLastUseContract:
    def test_in_use_retire_requires_last_use_event(self):
        state = ChunkTransportState(block_id=0)
        ticket = _drive_to_in_use(state)
        with pytest.raises(RuntimeError, match="without a last-use event"):
            state.record_last_use(ticket, None)

    def test_ready_retire_without_event_allowed(self):
        """A never-consumed tail prefetch may retire at producer-ready."""
        state = ChunkTransportState(block_id=0)
        ticket = _begin(state)
        state.mark_ready(ticket)
        state.record_last_use(ticket, None)
        assert state.slots[0].phase is SlotPhase.RETIRED

    def test_bound_retire_without_event_allowed(self):
        state = ChunkTransportState(block_id=0)
        ticket = _begin(state)
        state.mark_ready(ticket)
        state.mark_bound(ticket)
        state.record_last_use(ticket, None)
        assert state.slots[0].phase is SlotPhase.RETIRED


class TestFailureExits:
    def test_cancel_pre_collective(self):
        state = ChunkTransportState(block_id=0)
        ticket = _begin(state)
        state.cancel(ticket, "pin memory failed")
        assert state.slots[0].phase is SlotPhase.CANCELLED
        assert state.slots[0].error == "pin memory failed"
        assert not state.is_current(ticket)

    def test_begin_after_cancel_allowed(self):
        state = ChunkTransportState(block_id=0)
        ticket = _begin(state)
        state.cancel(ticket, "pin memory failed")
        retry = _begin(state, part_id="attention")
        assert retry.forward_generation == ticket.forward_generation + 1
        assert state.slots[0].phase is SlotPhase.PREFETCH_ENQUEUED

    def test_cancel_after_first_collective_rejected(self):
        state = ChunkTransportState(block_id=0)
        ticket = _begin(state)
        state.mark_ready(ticket)
        with pytest.raises(RuntimeError, match="use poison"):
            state.cancel(ticket, "too late")

    def test_poison_marks_live_slots_and_blocks_everything(self):
        state = ChunkTransportState(block_id=0)
        ticket = _drive_to_in_use(state, slot=0)
        _begin(state, slot=1, part_id="moe")
        state.poison("hccl error mid allgather")

        assert state.poisoned == "hccl error mid allgather"
        assert state.slots[0].phase is SlotPhase.POISONED_PROCESS_GROUP
        assert state.slots[1].phase is SlotPhase.POISONED_PROCESS_GROUP

        with pytest.raises(RuntimeError, match="process group is poisoned"):
            _begin(state, slot=0, request_generation=1)
        with pytest.raises(RuntimeError, match="process group is poisoned"):
            state.mark_ready(ticket)
        with pytest.raises(RuntimeError, match="cannot reset a poisoned"):
            state.reset()
        with pytest.raises(RuntimeError, match="cannot close a poisoned"):
            state.close()

    def test_reset_with_in_flight_slot_rejected(self):
        for phase in ("enqueued", "ready", "bound", "in_use"):
            fresh = ChunkTransportState(block_id=0)
            t = _begin(fresh)
            if phase in ("ready", "bound", "in_use"):
                fresh.mark_ready(t)
            if phase in ("bound", "in_use"):
                fresh.mark_bound(t)
            if phase == "in_use":
                fresh.mark_in_use(t)
            with pytest.raises(RuntimeError, match="in-flight"):
                fresh.reset()


# ---------------------------------------------------------------------- #
#  ChunkedWeightTransport wrapper                                        #
# ---------------------------------------------------------------------- #


def _manifest(block_id: int = 0, part_id: str = "attention", chunk_count: int = 2) -> PartManifest:
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


def _prepared_transport() -> ChunkedWeightTransport:
    transport = ChunkedWeightTransport(block_id=0, slot_count=2)
    transport.prepare(_manifest(), {torch.float32: torch.zeros(8)})
    return transport


class TestChunkedWeightTransport:
    def test_begin_submission_requires_prepare(self):
        transport = ChunkedWeightTransport(block_id=0)
        with pytest.raises(RuntimeError, match="not prepared"):
            transport.begin_submission(output_slot=0, request_generation=0, ready_event=object())

    def test_wrapper_lifecycle_and_counters(self):
        transport = _prepared_transport()
        ticket = transport.begin_submission(
            output_slot=0,
            request_generation=0,
            ready_event=object(),
            part_id="attention",
            backend_id="reference",
            last_collective_key=_KEY,
        )
        assert transport.counters.submissions == 1
        assert transport.counters.submitted_chunks == ticket.chunk_count

        # Idempotent re-submission with the same key does not double-count.
        again = transport.begin_submission(
            output_slot=0,
            request_generation=0,
            ready_event=object(),
            part_id="attention",
            backend_id="reference",
            last_collective_key=_KEY,
        )
        assert again is ticket
        assert transport.counters.submissions == 1

        transport.mark_ready(ticket)

        waited: list[object] = []
        with pytest.raises(RuntimeError, match="cannot attach consumer"):
            transport.attach_ready(ticket, waited.append)

        transport.mark_bound(ticket)
        transport.attach_ready(ticket, waited.append)
        assert waited == [ticket.ready_event]
        assert transport.counters.consumer_attaches == 1

        transport.record_last_use(ticket, object())
        assert transport.counters.releases == 1
        assert transport.state.slots[0].phase is SlotPhase.RETIRED

        transport.confirm_reusable(ticket)
        assert transport.state.slots[0].phase is SlotPhase.REUSABLE

        transport.reset()
        assert transport.counters.resets == 1

    def test_cancel_submission_counts_and_frees_slot(self):
        transport = _prepared_transport()
        ticket = transport.begin_submission(
            output_slot=0,
            request_generation=0,
            ready_event=object(),
            backend_id="reference",
            last_collective_key=_KEY,
        )
        transport.cancel_submission(ticket, "pin failed")
        assert transport.counters.cancels == 1
        retry = transport.begin_submission(
            output_slot=0,
            request_generation=0,
            ready_event=object(),
            backend_id="reference",
            last_collective_key=_KEY,
        )
        assert retry is not ticket

    def test_poison_counts_and_blocks_reuse(self):
        transport = _prepared_transport()
        transport.begin_submission(
            output_slot=0,
            request_generation=0,
            ready_event=object(),
            backend_id="reference",
            last_collective_key=_KEY,
        )
        transport.poison("mid-collective failure")
        assert transport.counters.poisons == 1
        with pytest.raises(RuntimeError, match="poisoned"):
            transport.begin_submission(output_slot=1, request_generation=0, ready_event=object())
