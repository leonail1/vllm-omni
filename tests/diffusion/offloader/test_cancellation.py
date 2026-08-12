# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for design section 27 hardening paths.

Covers the 27.2 poison registry / fail-closed marker and the 27.3
generation-bound cancellation channel (flag file + FS-group vote).

CPU-only: no streams, no collectives, no NPU required.
"""

from __future__ import annotations

import pytest

from vllm_omni.diffusion.offloader import cancellation
from vllm_omni.diffusion.offloader.cancellation import (
    check_cancellation,
    clear_current_request_ids,
    configure_abort_channel,
    sanitize_request_id,
    set_current_request_ids,
    write_abort_flag,
)
from vllm_omni.diffusion.offloader.chunked_transport import (
    DLO_POISON_ERROR_MARKER,
    ChunkTransportState,
    _clear_process_group_poison_for_tests,
    is_dlo_poison_error,
    process_group_poison_reason,
    record_process_group_poison,
)


@pytest.fixture(autouse=True)
def _clean_state(tmp_path, monkeypatch):
    cancellation._reset_for_tests()
    _clear_process_group_poison_for_tests()
    monkeypatch.setenv(cancellation.DLO_ABORT_DIR_ENV, str(tmp_path))
    yield
    cancellation._reset_for_tests()
    _clear_process_group_poison_for_tests()


class TestPoisonRegistry:
    def test_poison_records_process_wide_reason(self):
        state = ChunkTransportState(block_id=0)
        assert process_group_poison_reason() is None
        state.poison("hccl error mid allgather")
        assert process_group_poison_reason() == "hccl error mid allgather"

    def test_first_poison_reason_wins(self):
        record_process_group_poison("first")
        record_process_group_poison("second")
        assert process_group_poison_reason() == "first"

    def test_marker_detection(self):
        assert is_dlo_poison_error(f"{DLO_POISON_ERROR_MARKER}: boom")
        assert is_dlo_poison_error(f"Worker failed with error '{DLO_POISON_ERROR_MARKER}: boom'")
        assert not is_dlo_poison_error("ordinary error")
        assert not is_dlo_poison_error(None)


class TestExecutorFailClosed:
    def test_poison_error_fails_closed(self):
        from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor

        executor = object.__new__(MultiprocDiffusionExecutor)
        executor._is_failed = False
        executor._failure_callbacks = []
        executor.shutdown = lambda: None  # type: ignore[method-assign]

        executor._maybe_fail_closed_on_error_text("ordinary error")
        assert not executor._is_failed

        executor._maybe_fail_closed_on_error_text(f"{DLO_POISON_ERROR_MARKER}: mid allgather")
        assert executor._is_failed


class TestCancellationChannel:
    def test_unconfigured_channel_is_noop(self):
        check_cancellation()

    def test_no_request_ids_is_noop(self, tmp_path):
        configure_abort_channel(str(tmp_path), None, 1)
        check_cancellation()

    def test_no_flag_passes(self, tmp_path):
        configure_abort_channel(str(tmp_path), None, 1)
        set_current_request_ids(["req-1"])
        check_cancellation()

    def test_flag_raises_and_clears_per_request(self, tmp_path):
        configure_abort_channel(str(tmp_path), None, 1)
        set_current_request_ids(["req-1"])
        write_abort_flag("req-1")
        with pytest.raises(RuntimeError, match="aborted by client"):
            check_cancellation()
        # Generation boundary: clearing ids (end_request) silences the flag;
        # the next request with a different id is unaffected.
        clear_current_request_ids()
        check_cancellation()
        set_current_request_ids(["req-2"])
        check_cancellation()

    def test_flag_of_other_request_ignored(self, tmp_path):
        configure_abort_channel(str(tmp_path), None, 1)
        set_current_request_ids(["req-1"])
        write_abort_flag("req-999")
        check_cancellation()

    def test_request_id_sanitized(self, tmp_path):
        configure_abort_channel(str(tmp_path), None, 1)
        weird = "video/sync:abc 123"
        set_current_request_ids([weird])
        write_abort_flag(weird)
        assert sanitize_request_id(weird) not in ("", weird)
        with pytest.raises(RuntimeError, match="aborted by client"):
            check_cancellation()

    def test_vote_uses_group_when_multi_rank(self, tmp_path):
        """Multi-rank path must all_reduce the vote even when the local flag
        is absent (uniformity proof happens in the gloo dist test)."""
        configure_abort_channel(str(tmp_path), None, 2)
        set_current_request_ids(["req-1"])
        # torch.distributed is not initialized in this process: the vote must
        # fall back to the local flag instead of raising.
        check_cancellation()
