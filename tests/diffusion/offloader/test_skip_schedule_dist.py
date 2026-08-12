# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Multi-process fault-injection tests for the section-24 policy-2 digest sync.

Two CPU processes form a gloo group standing in for the FS group.  Both run
the real ``SkipScheduleCoordinator``; only the skip *decision sequences* are
scripted, which is exactly the fault policy 2 guards against.

Verified properties:

* agreement: identical decision sequences never raise;
* divergence: a single-rank divergent decision makes BOTH ranks raise the
  same digest-mismatch error at the same decision index (no split-brain);
* clean failure: after the mismatch the process group is still usable —
  a fresh coordinator on the same group completes further compares
  (pre-collective failure, design section 27.1, not a poisoned group).
"""

from __future__ import annotations

import datetime
import os

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_omni.diffusion.offloader.skip_schedule import (
    BlockPartSchedule,
    SkipScheduleCoordinator,
)

_WORLD_SIZE = 2


def _blocks() -> list[BlockPartSchedule]:
    return [
        BlockPartSchedule(
            block_id=i,
            parts=(
                ("attention", (("torch.bfloat16", 3),)),
                ("moe", (("torch.bfloat16", 5),)),
            ),
        )
        for i in range(4)
    ]


def _feed(coordinator: SkipScheduleCoordinator, decisions: list[bool]) -> str | None:
    """Feed scripted decisions; return the mismatch error message, if any."""
    try:
        for skipped in decisions:
            coordinator.on_decision(skipped=skipped, fn_blocks=1, bn_blocks=0)
    except RuntimeError as exc:
        return str(exc)
    return None


def _worker(rank: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        group = dist.new_group(list(range(_WORLD_SIZE)), backend="gloo")

        def make_coordinator(generation: int) -> SkipScheduleCoordinator:
            coordinator = SkipScheduleCoordinator(
                domain="gen_layers",
                blocks=_blocks(),
                backend_id="reference",
                cpu_group=group,
                group_size=_WORLD_SIZE,
            )
            coordinator.set_request_generation(generation)
            return coordinator

        # 1. Agreement: identical decision sequences never raise.
        coordinator = make_coordinator(0)
        error = _feed(coordinator, [False, True, False, True])
        assert error is None, f"agreement run raised: {error}"
        assert coordinator.counters.digest_compares == 4

        # 2. Divergence: rank 0 skips at decision index 1, rank 1 computes.
        #    Both ranks must fail with the same digest-mismatch error.
        coordinator = make_coordinator(1)
        divergent = [False, True] if rank == 0 else [False, False]
        error = _feed(coordinator, divergent)
        assert error is not None, "divergent decision was not detected"
        assert "digest mismatch" in error, error
        assert "decision_index=1" in error, error

        # 3. The failure is pre-collective (27.1): the group is NOT poisoned
        #    and immediately serves another collective.
        token = [None] * _WORLD_SIZE
        dist.all_gather_object(token, f"rank-{rank}-alive", group=group)
        assert token == ["rank-0-alive", "rank-1-alive"]

        # 4. Post-incident agreement on the same group works: the next
        #    request proceeds normally.
        coordinator = make_coordinator(2)
        error = _feed(coordinator, [True, False])
        assert error is None, f"post-incident run raised: {error}"
        assert coordinator.counters.digest_compares == 2

        # 5. Cancellation vote (design section 27.3): only rank 0 can see the
        #    flag file, but the FS-group vote must abort BOTH ranks at the
        #    same boundary.
        from vllm_omni.diffusion.offloader import cancellation
        from vllm_omni.diffusion.offloader.cancellation import (
            check_cancellation,
            configure_abort_channel,
            set_current_request_ids,
        )

        abort_dir = os.path.join(os.path.dirname(init_file), "abort_flags")
        os.makedirs(abort_dir, exist_ok=True)
        cancellation._reset_for_tests()
        configure_abort_channel(abort_dir, group, _WORLD_SIZE)
        set_current_request_ids(["req-cancel"])
        if rank == 0:
            with open(os.path.join(abort_dir, "req-cancel"), "w"):
                pass
        # Rank 1 may reach the check before rank 0's write is visible... the
        # vote unifies either way: both must raise.
        raised = False
        try:
            check_cancellation()
        except RuntimeError as exc:
            assert "aborted by client" in str(exc)
            raised = True
        assert raised, f"rank {rank} did not abort on the cancellation vote"
        cancellation._reset_for_tests()

        dist.barrier(group=group)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="gloo is required")
def test_skip_schedule_digest_sync_fault_injection(tmp_path):
    init_file = tmp_path / "gloo_init"
    mp.spawn(_worker, args=(str(init_file),), nprocs=_WORLD_SIZE, join=True)
