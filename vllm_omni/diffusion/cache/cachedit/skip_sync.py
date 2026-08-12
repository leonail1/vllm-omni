# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""cache-dit skip-decision observer (design section 24, policy 2 adapter).

cache-dit's DBCache decides once per ``CachedBlocks.forward`` — after the
leading Fn blocks computed, before the remaining blocks run — whether the
rest of the step is served from cache.  That decision point is exactly where
the DLO part pipeline needs to prove FS-group agreement before the first
decision-dependent weight collective.

This module wraps ``CachedContextManager.can_cache`` (the single chokepoint
every forward pattern calls) and forwards the returned decision to the
offloader's skip-schedule registry.  The wrapper is installed once, at
``CacheDiTBackend.enable`` time; with no registered coordinator the
notification is a no-op, so cache-dit without the part pipeline is
unaffected.

The wrapper intentionally depends on cache-dit internals; it was written
against cache-dit 1.3.x and fails loudly at install time if the expected
hook point is missing, rather than silently skipping the agreement proof.
"""

from __future__ import annotations

import functools
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.offloader.skip_schedule import (
    mark_skip_decision_observer_installed,
    notify_skip_decision,
)

logger = init_logger(__name__)

_OBSERVER_INSTALLED = False
_ORIGINAL_CAN_CACHE: Any | None = None


def install_skip_decision_observer() -> None:
    """Wrap cache-dit's ``can_cache`` to observe every skip decision.

    Idempotent.  Raises RuntimeError when the installed cache-dit version
    does not expose the expected decision point.
    """
    global _OBSERVER_INSTALLED, _ORIGINAL_CAN_CACHE
    if _OBSERVER_INSTALLED:
        # The offloader-side registry flag may have been reset independently
        # (e.g. by test fixtures); keep the two in sync.
        mark_skip_decision_observer_installed()
        return

    try:
        from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager
    except ImportError as exc:
        raise RuntimeError(
            "cache-dit skip-decision observer requires cache-dit with "
            "caching.cache_contexts.cache_manager.CachedContextManager; "
            "the installed cache-dit version is not supported"
        ) from exc

    original = getattr(CachedContextManager, "can_cache", None)
    if original is None or not callable(original):
        raise RuntimeError(
            "cache-dit skip-decision observer: CachedContextManager.can_cache is missing; "
            "the installed cache-dit version is not supported"
        )

    @functools.wraps(original)
    def _observed_can_cache(self, states_tensor, parallelized=False, threshold=None, prefix="Fn"):
        decision = bool(original(self, states_tensor, parallelized=parallelized, threshold=threshold, prefix=prefix))
        # Fn/Bn shape the decision-dependent block range in the schedule
        # digest; fall back to the vLLM-Omni defaults when the context does
        # not expose them.
        try:
            fn_blocks = int(self.Fn_compute_blocks)
            bn_blocks = int(self.Bn_compute_blocks)
        except Exception:
            fn_blocks, bn_blocks = 1, 0
        notify_skip_decision(skipped=decision, prefix=prefix, fn_blocks=fn_blocks, bn_blocks=bn_blocks)
        return decision

    CachedContextManager.can_cache = _observed_can_cache
    _ORIGINAL_CAN_CACHE = original
    _OBSERVER_INSTALLED = True
    mark_skip_decision_observer_installed()
    logger.info("Installed cache-dit skip-decision observer (design section 24, policy 2)")


def skip_decision_observer_active() -> bool:
    return _OBSERVER_INSTALLED
