# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded receive buffers for sequential inference on one compute stream.

The caller must join every collective Work on that stream before release().
This cache neither joins communication nor permits simultaneous leases. It is
not safe for graph capture, concurrent requests, or a different compute stream.
"""

from collections import OrderedDict
from contextlib import contextmanager


def _record_buffers(buffers, stream):
    if isinstance(buffers, (list, tuple)):
        for buffer in buffers:
            _record_buffers(buffer, stream)
    else:
        buffers.record_stream(stream)


class HeadWorkspaceCache:
    def __init__(self, capacity=2):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Capacity must be a positive integer")
        self.capacity = capacity
        self._entries = OrderedDict()
        self._stream_key = None
        self._active_key = None
        self._leased = False
        self._poisoned = False
        self._closed = False

    def acquire(self, key, allocate, *, stream_key):
        """Return buffers, rejecting reentry and changes of compute stream.

        stream_key must identify device and native stream, not a transient
        Python wrapper. Allocation must run on that same compute stream.
        """
        if self._closed:
            raise RuntimeError("Workspace has been closed")
        if self._poisoned:
            raise RuntimeError("Failed use requires synchronized teardown")
        if self._leased:
            raise RuntimeError("Workspace is already leased")
        if self._stream_key is not None and stream_key != self._stream_key:
            raise RuntimeError("Workspace requires one compute stream")
        if key not in self._entries:
            if len(self._entries) == self.capacity:
                self._entries.popitem(last=False)
            self._entries[key] = allocate()
        self._entries.move_to_end(key)
        self._stream_key = stream_key
        self._active_key = key
        self._leased = True
        return self._entries[key]

    def release_after_join(self, stream, *, stream_key):
        """After ALL Work.wait calls on stream, register buffer lifetimes.

        Wait must establish device completion dependencies on the same compute
        stream. record_stream covers pending compute reads AND preceding joined
        communication when Python references are later evicted. It does not
        substitute for Work.wait. The caller is responsible for this ordering.
        """
        if not self._leased:
            raise RuntimeError("No active workspace")
        if stream_key != self._stream_key:
            raise RuntimeError("Release must join the acquisition stream")
        _record_buffers(self._entries[self._active_key], stream)
        self._active_key = None
        self._leased = False

    def abandon(self):
        """Preserve references after an exception with possibly unjoined work."""
        self._poisoned = True

    def clear_after_synchronize(self):
        """Caller must synchronize the device first, including exception paths."""
        self._entries.clear()
        self._active_key = None
        self._leased = False
        self._poisoned = False
        self._stream_key = None

    @property
    def size(self):
        return len(self._entries)

    @contextmanager
    def lease(self, key, allocate, *, stream, stream_key):
        """Body must join all collective work before returning normally.

        Exceptions retain references and prohibit reuse until synchronized
        teardown. Returns inside the body still execute lifetime registration.
        """
        buffers = self.acquire(key, allocate, stream_key=stream_key)
        try:
            yield buffers
            self.release_after_join(stream, stream_key=stream_key)
        except BaseException:
            self.abandon()
            raise

    def close_after_synchronize(self):
        """Permanently reject stale adapter forwards after model teardown."""
        self.clear_after_synchronize()
        self._closed = True
