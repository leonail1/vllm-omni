# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Collective-schedule digest sync for cache-dit dynamic block skip.

Design section 24, policy 2: when the Stage-3 part pipeline runs with
cache-dit's dynamic block skip enabled, a skipped step submits none of the
part collectives for the skipped blocks, so the FS group must prove — not
assume — that every rank made the same skip decision.  After each skip
decision, every rank derives the ``CollectiveSchedule`` (the canonical
collective keys it will still submit for this step's remaining blocks),
hashes it to a fixed-size digest, and compares digests across the FS group
before the first decision-dependent weight collective.

A digest mismatch is a pre-collective failure (design section 27.1): no
decision-dependent collective has been submitted yet, so every rank raises
the same error at the same point, the request fails fast, and the process
group stays healthy.

This module deliberately has no platform-stream dependency; the digest
compare runs on the FS CPU (gloo) process group, exactly like the startup
manifest consistency validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from .chunked_transport import metrics_enabled

logger = init_logger(__name__)

# Test-only failure injection (design section 30.6): when set, the
# coordinator records a process-group poison and raises at this decision
# index, exercising the section-27.2 fail-closed chain end to end.
_TEST_POISON_AT_DECISION = os.environ.get("VLLM_OMNI_DLO_TEST_POISON_AT_DECISION")

# Fixed digest size reported to the FS group (design section 24: "固定大小
# digest").  16 hex chars of sha256 keeps the all_gather payload tiny while
# making accidental collisions negligible.
DIGEST_HEX_CHARS = 16


@dataclass(frozen=True)
class BlockPartSchedule:
    """Static per-block collective template for one streamed block.

    Attributes:
        block_id: Runtime block id in execution order.
        parts: ``(part_id, ((dtype_name, chunk_count), ...))`` per part, in
            submission order.  Expanded into canonical collective keys for
            every step in which the block is scheduled to run.
    """

    block_id: int
    parts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]


def build_step_schedule_digest(
    *,
    request_generation: int,
    decision_index: int,
    skipped: bool,
    blocks: tuple[BlockPartSchedule, ...],
    fn_blocks: int,
    bn_blocks: int,
    backend_id: str,
) -> str:
    """Hash this step's decision-dependent collective key sequence.

    The first ``fn_blocks`` blocks (and the last ``bn_blocks``) execute
    unconditionally, so their collectives are identical across ranks by
    construction and are excluded.  For every remaining block, a compute
    step submits both parts' chunk collectives; a skipped step submits none.
    """
    if fn_blocks < 0 or bn_blocks < 0 or fn_blocks + bn_blocks > len(blocks):
        raise ValueError(f"invalid Fn/Bn for {len(blocks)} blocks: fn_blocks={fn_blocks}, bn_blocks={bn_blocks}")
    end = len(blocks) - bn_blocks if bn_blocks else len(blocks)
    affected = blocks[fn_blocks:end]

    keys: list[list[Any]] = []
    if not skipped:
        for block in affected:
            for part_id, dtypes in block.parts:
                for dtype_name, chunk_count in dtypes:
                    for chunk_id in range(chunk_count):
                        keys.append([block.block_id, part_id, dtype_name, chunk_id, backend_id])

    payload = {
        "request_generation": request_generation,
        "decision_index": decision_index,
        "skipped": skipped,
        "keys": keys,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[
        :DIGEST_HEX_CHARS
    ]


@dataclass
class SkipScheduleCounters:
    decisions: int = 0
    skipped: int = 0
    digest_compares: int = 0


class SkipScheduleCoordinator:
    """Owns the per-decision digest compare for one DiT block group.

    One coordinator serves one cache-dit decision domain (one ``CachedBlocks``
    wrapper).  ``on_decision`` is invoked from the cache-dit decision point —
    after the leading Fn blocks computed and before the first
    decision-dependent weight collective — on every FS rank, in the same
    order, so the all_gather below never deadlocks.
    """

    def __init__(
        self,
        *,
        domain: str,
        blocks: list[BlockPartSchedule],
        backend_id: str,
        cpu_group: Any | None,
        group_size: int,
        on_skip: Any | None = None,
    ) -> None:
        if not domain:
            raise ValueError("skip schedule coordinator requires a decision domain name")
        if not blocks:
            raise ValueError("skip schedule coordinator requires at least one block")
        self.domain = domain
        self._blocks = tuple(sorted(blocks, key=lambda block: block.block_id))
        self._backend_id = backend_id
        self._cpu_group = cpu_group
        self._group_size = group_size
        # Invoked after a SUCCESSFUL digest compare when the decision is
        # "skip": lets the part pipeline retire the pre-decision prefetch
        # whose data cannot survive until the next compute step.
        self._on_skip = on_skip
        self._request_generation = 0
        self._decision_index = 0
        self.counters = SkipScheduleCounters()
        # Cached at construction: counter updates cost one predictable
        # branch per decision when the metrics gate is off.
        self._metrics_on = metrics_enabled()
        self.last_digest: str | None = None

    def set_request_generation(self, generation: int) -> None:
        if generation < self._request_generation:
            raise RuntimeError(f"request generation moved backwards: {generation} < {self._request_generation}")
        self._request_generation = generation
        self._decision_index = 0

    def on_decision(self, *, skipped: bool, fn_blocks: int, bn_blocks: int) -> None:
        """Compare this rank's step schedule digest against the FS group.

        Raises RuntimeError on mismatch (design section 27.1 fail-fast): the
        request is aborted by every rank at the same point and the process
        group is left healthy.
        """
        digest = build_step_schedule_digest(
            request_generation=self._request_generation,
            decision_index=self._decision_index,
            skipped=skipped,
            blocks=self._blocks,
            fn_blocks=fn_blocks,
            bn_blocks=bn_blocks,
            backend_id=self._backend_id,
        )
        self._decision_index += 1
        if self._metrics_on:
            self.counters.decisions += 1
            self.counters.skipped += int(skipped)
        self.last_digest = digest

        if _TEST_POISON_AT_DECISION is not None and self._decision_index - 1 == int(_TEST_POISON_AT_DECISION):
            from .chunked_transport import record_process_group_poison

            record_process_group_poison(
                f"test-injected fault at decision {self._decision_index - 1} (domain={self.domain!r})"
            )
            raise RuntimeError(
                f"test-injected mid-collective failure at decision {self._decision_index - 1} "
                f"(domain={self.domain!r}, generation={self._request_generation})"
            )
        logger.debug(
            "skip-schedule decision: domain=%s gen=%s idx=%s skipped=%s digest=%s",
            self.domain,
            self._request_generation,
            self._decision_index - 1,
            skipped,
            digest,
        )

        if self._group_size <= 1 or self._cpu_group is None or not torch.distributed.is_initialized():
            if skipped and self._on_skip is not None:
                self._on_skip()
            return

        gathered: list[Any] = [None] * self._group_size
        torch.distributed.all_gather_object(gathered, digest, group=self._cpu_group)
        if self._metrics_on:
            self.counters.digest_compares += 1
        mismatched = [rank for rank, remote in enumerate(gathered) if remote != digest]
        if mismatched:
            raise RuntimeError(
                "cache-dit skip decision digest mismatch across the FS group: "
                f"domain={self.domain!r} request_generation={self._request_generation} "
                f"decision_index={self._decision_index - 1} skipped={skipped} "
                f"mismatched_ranks={mismatched} — aborting the request before any "
                "decision-dependent collective (design sections 24/27.1)"
            )
        if skipped and self._on_skip is not None:
            self._on_skip()


# ---------------------------------------------------------------------- #
#  Registry: how the cache-dit observer reaches the active coordinators   #
# ---------------------------------------------------------------------- #
#
# The cache-dit decision observer (vllm_omni/diffusion/cache/cachedit/
# skip_sync.py) wraps cache-dit's decision function and forwards every
# decision here.  The DLO backend registers one coordinator per streamed
# block group, keyed by the group's blocks-container name (e.g.
# "gen_layers") — cache-dit's decision prefix is "<blocks_name>_Fn_*", so a
# decision routes to the coordinator whose domain prefixes it.  Groups that
# cache-dit never wraps (e.g. a run-once text pathway) simply receive no
# decisions.  With no registered coordinator, notifications are a no-op —
# cache-dit without the part pipeline needs no schedule proof.

_COORDINATORS: dict[str, SkipScheduleCoordinator] = {}
_OBSERVER_INSTALLED = False
_REGISTRY_LOCK = threading.Lock()
_UNMATCHED_PREFIXES: set[str] = set()


def register_skip_schedule_coordinator(domain: str, coordinator: SkipScheduleCoordinator) -> None:
    with _REGISTRY_LOCK:
        if domain in _COORDINATORS:
            raise RuntimeError(f"skip schedule coordinator already registered for domain {domain!r}")
        if coordinator.domain != domain:
            raise RuntimeError(
                f"coordinator domain {coordinator.domain!r} does not match registration domain {domain!r}"
            )
        _COORDINATORS[domain] = coordinator


def unregister_skip_schedule_coordinator(coordinator: SkipScheduleCoordinator) -> None:
    with _REGISTRY_LOCK:
        if _COORDINATORS.get(coordinator.domain) is coordinator:
            del _COORDINATORS[coordinator.domain]


def has_skip_schedule_coordinator() -> bool:
    with _REGISTRY_LOCK:
        return bool(_COORDINATORS)


def mark_skip_decision_observer_installed() -> None:
    global _OBSERVER_INSTALLED
    with _REGISTRY_LOCK:
        _OBSERVER_INSTALLED = True


def skip_decision_observer_installed() -> bool:
    with _REGISTRY_LOCK:
        return _OBSERVER_INSTALLED


def notify_skip_decision(*, skipped: bool, prefix: str, fn_blocks: int, bn_blocks: int) -> None:
    """Route one cache-dit skip decision to the matching coordinator.

    The decision prefix is "<blocks_name>_Fn_residual" (or the hidden-states
    variant); the owning coordinator is the one whose domain the prefix
    starts with.  An unmatched prefix means cache-dit wrapped a block list
    the part pipeline does not stream (no weight collectives to protect), so
    it is logged once and ignored.
    """
    with _REGISTRY_LOCK:
        coordinators = dict(_COORDINATORS)
    # Longest domain first so "gen_layers" beats "gen" if both exist.
    for domain in sorted(coordinators, key=len, reverse=True):
        if prefix == domain or prefix.startswith(f"{domain}_"):
            coordinators[domain].on_decision(skipped=skipped, fn_blocks=fn_blocks, bn_blocks=bn_blocks)
            return
    with _REGISTRY_LOCK:
        if prefix not in _UNMATCHED_PREFIXES:
            _UNMATCHED_PREFIXES.add(prefix)
            logger.warning(
                "cache-dit skip decision for prefix %r has no registered DLO skip-schedule "
                "coordinator; if these blocks are part-pipeline streamed, their weight "
                "collectives are NOT covered by the section-24 policy-2 digest sync",
                prefix,
            )


def _reset_registry_for_tests() -> None:
    global _OBSERVER_INSTALLED
    with _REGISTRY_LOCK:
        _COORDINATORS.clear()
        _UNMATCHED_PREFIXES.clear()
        _OBSERVER_INSTALLED = False
