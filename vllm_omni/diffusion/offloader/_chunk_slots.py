# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-slot transfer tickets: REUSABLE → SUBMITTED → READY → IN_USE → REUSABLE."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from ._chunk_types import PartManifest, SlotPhase


@dataclass
class TransferTicket:
    request_generation: int
    forward_generation: int
    block_id: int
    part_id: str
    output_slot: int
    chunk_count: int
    ready_event: Any | None = field(default=None, compare=False)
    last_collective_key: tuple[Any, ...] | None = None


@dataclass
class OutputSlotState:
    phase: SlotPhase = SlotPhase.REUSABLE
    ticket: TransferTicket | None = None
    last_use_event: Any | None = None


class ChunkTransportState:
    def __init__(self, block_id: int, slot_count: int = 2) -> None:
        self.block_id = block_id
        self._forward_generation = 0
        self._slots = [OutputSlotState() for _ in range(slot_count)]
        self._closed = False

    @property
    def slots(self) -> tuple[OutputSlotState, ...]:
        return tuple(self._slots)

    def begin(
        self, *, output_slot: int, chunk_count: int, request_generation: int = 0,
        part_id: str = "block", ready_event: Any | None = None,
        last_collective_key: tuple[Any, ...] | None = None,
        backend_id: str | None = None,
    ) -> TransferTicket:
        del backend_id
        if self._closed:
            raise RuntimeError("chunk transport state is closed")
        slot = self._slots[output_slot]
        if slot.phase is not SlotPhase.REUSABLE:
            cur = slot.ticket
            same = (
                cur is not None and cur.request_generation == request_generation
                and cur.block_id == self.block_id and cur.part_id == part_id
                and cur.chunk_count == chunk_count and cur.last_collective_key == last_collective_key
            )
            if same:
                return cur  # type: ignore[return-value]
            owned = (
                (cur.request_generation, cur.forward_generation, cur.block_id, cur.part_id)
                if cur else slot.phase.value
            )
            raise RuntimeError(f"output slot {output_slot} is owned by {owned}")
        self._forward_generation += 1
        ticket = TransferTicket(
            request_generation, self._forward_generation, self.block_id, part_id,
            output_slot, chunk_count, ready_event, last_collective_key,
        )
        slot.phase, slot.ticket = SlotPhase.SUBMITTED, ticket
        return ticket

    def mark_ready(self, ticket: TransferTicket) -> None:
        slot = self._require_current(ticket)
        if slot.phase is not SlotPhase.SUBMITTED:
            raise RuntimeError(f"cannot mark {slot.phase.value} slot ready")
        slot.phase = SlotPhase.READY

    def mark_in_use(self, ticket: TransferTicket) -> None:
        slot = self._require_current(ticket)
        if slot.phase not in (SlotPhase.READY, SlotPhase.IN_USE):
            raise RuntimeError(f"cannot attach consumer to {slot.phase.value} slot")
        slot.phase = SlotPhase.IN_USE

    def release(self, ticket: TransferTicket, last_use_event: Any | None) -> None:
        slot = self._require_current(ticket)
        if slot.phase not in (SlotPhase.READY, SlotPhase.IN_USE):
            raise RuntimeError(f"cannot release {slot.phase.value} slot")
        slot.last_use_event, slot.ticket, slot.phase = last_use_event, None, SlotPhase.REUSABLE

    def is_current(self, ticket: TransferTicket) -> bool:
        return self._slots[ticket.output_slot].ticket == ticket

    def reset(self) -> None:
        if any(s.phase in (SlotPhase.SUBMITTED, SlotPhase.IN_USE) for s in self._slots):
            raise RuntimeError("cannot reset chunk transport with in-flight slots")
        self._slots = [OutputSlotState() for _ in self._slots]
        self._forward_generation = 0
        self._closed = False

    def close(self) -> None:
        if any(s.phase in (SlotPhase.SUBMITTED, SlotPhase.IN_USE) for s in self._slots):
            raise RuntimeError("cannot close chunk transport with in-flight slots")
        self._closed = True

    def _require_current(self, ticket: TransferTicket) -> OutputSlotState:
        slot = self._slots[ticket.output_slot]
        if slot.ticket != ticket:
            raise RuntimeError(f"stale transfer ticket for output slot {ticket.output_slot}")
        return slot


@dataclass
class TransportCounters:
    submissions: int = 0
    submitted_chunks: int = 0
    consumer_attaches: int = 0
    releases: int = 0
    resets: int = 0


class ChunkedWeightTransport(ChunkTransportState):
    def __init__(self, block_id: int, slot_count: int = 2) -> None:
        super().__init__(block_id, slot_count=slot_count)
        self.state = self
        self.manifest: PartManifest | None = None
        self.host_shards: dict[torch.dtype, torch.Tensor] = {}
        self.counters = TransportCounters()

    def prepare(self, manifest: PartManifest, host_shards: dict[torch.dtype, torch.Tensor]) -> None:
        if manifest.block_id != self.block_id:
            raise ValueError(
                f"manifest block_id={manifest.block_id} does not match transport block_id={self.block_id}"
            )
        expected = {d.dtype: d.local_numel for d in manifest.dtypes}
        actual = {dtype: shard.numel() for dtype, shard in host_shards.items()}
        if actual != expected:
            raise ValueError(f"Host shard sizes do not match manifest: expected={expected}, actual={actual}")
        self.manifest, self.host_shards = manifest, host_shards

    def begin_submission(
        self, *, output_slot: int, request_generation: int, ready_event: Any,
        part_id: str = "block", last_collective_key: tuple[Any, ...] | None = None,
        backend_id: str | None = None,
    ) -> TransferTicket:
        if self.manifest is None:
            raise RuntimeError("chunk transport is not prepared")
        previous = self.slots[output_slot].ticket
        ticket = self.begin(
            output_slot=output_slot, chunk_count=self.manifest.chunk_count,
            request_generation=request_generation, part_id=part_id,
            ready_event=ready_event, last_collective_key=last_collective_key,
            backend_id=backend_id,
        )
        if ticket is not previous:
            self.counters.submissions += 1
            self.counters.submitted_chunks += ticket.chunk_count
        return ticket

    def attach_ready(self, ticket: TransferTicket, wait_event: Callable[[Any], None]) -> None:
        self.mark_in_use(ticket)
        wait_event(ticket.ready_event)
        self.counters.consumer_attaches += 1

    def record_last_use(self, ticket: TransferTicket, last_use_event: Any) -> None:
        self.release(ticket, last_use_event)
        self.counters.releases += 1

    def reset(self) -> None:
        super().reset()
        self.counters.resets += 1

    def reset_counters(self) -> None:
        self.counters = TransportCounters()

    def close(self) -> None:
        super().close()
        self.host_shards = {}
