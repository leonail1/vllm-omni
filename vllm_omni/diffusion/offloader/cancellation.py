# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generation-bound request cancellation for the DLO chunk engine.

Design section 27.3: a client disconnect must never let one FS rank exit a
forward alone.  Workers cannot receive RPCs mid-forward, so the engine
proc signals an abort through a per-request flag file; every rank checks
the flag at the same forward boundary (the group-first block's
``pre_forward``) and then votes on the FS CPU group.  The vote makes the
decision uniform: either every rank aborts the pass at the same boundary
or none does, so the collective sequence never diverges.

Cancellation is bound to the request generation: the aborting request's
slots are cleaned up by the normal ``end_request``/``drain_request`` path,
which only touches tickets of the current generation.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Sequence
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Environment variable carrying the per-engine abort-flag directory.  The
# executor sets it before spawning workers (spawn children inherit it) and
# the engine proc reads it when writing flag files.
DLO_ABORT_DIR_ENV = "VLLM_OMNI_DLO_ABORT_DIR"

_LOCK = threading.Lock()
_ABORT_DIR: str | None = None
_CPU_GROUP: Any | None = None
_GROUP_SIZE: int = 1
_CURRENT_REQUEST_IDS: tuple[str, ...] = ()

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_request_id(request_id: str) -> str:
    return _SAFE_NAME.sub("_", request_id)


def configure_abort_channel(abort_dir: str | None, cpu_group: Any | None, group_size: int) -> None:
    """Enable cancellation checks for this worker's FS group."""
    global _ABORT_DIR, _CPU_GROUP, _GROUP_SIZE
    with _LOCK:
        _ABORT_DIR = abort_dir or None
        _CPU_GROUP = cpu_group
        _GROUP_SIZE = max(1, int(group_size))


def abort_channel_configured() -> bool:
    with _LOCK:
        return _ABORT_DIR is not None


def set_current_request_ids(request_ids: Sequence[str]) -> None:
    global _CURRENT_REQUEST_IDS
    with _LOCK:
        _CURRENT_REQUEST_IDS = tuple(request_ids)


def clear_current_request_ids() -> None:
    global _CURRENT_REQUEST_IDS
    with _LOCK:
        _CURRENT_REQUEST_IDS = ()


def write_abort_flag(request_id: str) -> None:
    """Engine side: signal that *request_id* was aborted by the client."""
    abort_dir = os.environ.get(DLO_ABORT_DIR_ENV)
    if not abort_dir:
        return
    try:
        flag = os.path.join(abort_dir, sanitize_request_id(request_id))
        with open(flag, "w"):
            pass
    except OSError:
        logger.warning("Failed to write DLO abort flag for request %s", request_id)


def check_cancellation() -> None:
    """Abort the current pass when the request was cancelled by the client.

    Called at the group-first block's pre_forward — the same forward
    boundary on every FS rank.  The FS-group vote (MAX reduce on a byte)
    turns per-rank flag visibility into a uniform decision, so all ranks
    raise together at the same boundary (design section 27.3).
    """
    with _LOCK:
        abort_dir = _ABORT_DIR
        request_ids = _CURRENT_REQUEST_IDS
        group = _CPU_GROUP
        group_size = _GROUP_SIZE
    if abort_dir is None or not request_ids:
        return

    vote = int(any(os.path.exists(os.path.join(abort_dir, sanitize_request_id(rid))) for rid in request_ids))
    if group_size > 1 and group is not None and torch.distributed.is_initialized():
        ballot = torch.tensor([vote], dtype=torch.int32)
        torch.distributed.all_reduce(ballot, op=torch.distributed.ReduceOp.MAX, group=group)
        vote = int(ballot.item())

    if vote:
        raise RuntimeError(
            f"DLO request {list(request_ids)} aborted by client: cancelling the forward at a "
            "uniform step boundary on every FS rank (design section 27.3)"
        )


def _reset_for_tests() -> None:
    global _ABORT_DIR, _CPU_GROUP, _GROUP_SIZE, _CURRENT_REQUEST_IDS
    with _LOCK:
        _ABORT_DIR = None
        _CPU_GROUP = None
        _GROUP_SIZE = 1
        _CURRENT_REQUEST_IDS = ()
