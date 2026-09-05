# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Failure injection for native resources and graph capture recovery."""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.distributed import sdma_transport as sdma


class RuntimeLibrary:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = []
        self.created = []
        self.functions = {}
        self.capturing = False

    def __getattr__(self, name):
        if name not in self.functions:

            def call(*args):
                self.calls.append((name, args))
                if self.failure == (name, sum(n == name for n, _ in self.calls)):
                    return 987
                if name in (
                    "aclrtMalloc",
                    "aclrtCreateNotify",
                    "aclrtNotifyImportByKey",
                    "aclrtIpcMemImportByKey",
                    "aclrtCreateStreamWithConfig",
                    "aclrtCreateEventExWithFlag",
                ):
                    value = 1000 + len(self.created)
                    args[0]._obj.value = value
                    self.created.append((name, value))
                elif name == "aclrtDeviceGetBareTgid":
                    args[0]._obj.value = 100
                elif name in ("aclrtIpcMemGetExportKey", "aclrtNotifyGetExportKey"):
                    args[2 if name == "aclrtIpcMemGetExportKey" else 1].value = b"export"
                elif name == "aclmdlRICaptureBegin":
                    assert not self.capturing
                    self.capturing = True
                elif name == "aclmdlRICaptureEnd":
                    assert self.capturing
                    self.capturing = False
                    args[1]._obj.value = 9999
                return 0

            self.functions[name] = call
        return self.functions[name]


@pytest.mark.parametrize("failure", [("aclrtCreateNotify", 3), ("aclrtNotifyImportByKey", 3)])
def test_partial_initialization_releases_created_resources(monkeypatch, failure):
    library = RuntimeLibrary(failure)
    monkeypatch.setattr(sdma.ct, "CDLL", lambda _: library)
    monkeypatch.setattr(sdma.torch, "npu", SimpleNamespace(current_device=lambda: 0), raising=False)
    monkeypatch.setattr(sdma.dist, "get_rank", lambda _: 0)
    monkeypatch.setattr(sdma.dist, "get_world_size", lambda _: 2)
    monkeypatch.setattr(sdma.dist, "get_process_group_ranks", lambda _: [0, 1])

    def exchange(self, value):
        if isinstance(value[0], int):
            return [(100, "host"), (101, "host")]
        return [value, (b"peer", [b"notify"] * 8)]

    monkeypatch.setattr(sdma.WeightTransport, "exchange", exchange)
    with pytest.raises(RuntimeError, match="returned 987"):
        sdma.WeightTransport(None, 4 * 1024**2)
    expected = {value for name, value in library.created if name != "aclrtIpcMemImportByKey"}
    released = [args[0] for name, args in library.calls if name in ("aclrtFree", "aclrtDestroyNotify")]
    assert len(released) == len(expected)
    assert set(released) == expected
    closed = [args[0] for name, args in library.calls if name == "aclrtIpcMemClose"]
    assert closed == ([b"peer", b"export"] if failure[0] == "aclrtNotifyImportByKey" else [b"export"])


def test_failed_capture_can_be_followed_by_another_capture(monkeypatch):
    library = RuntimeLibrary()
    monkeypatch.setattr(sdma.ct, "CDLL", lambda _: library)
    transport = sdma.WeightTransport.__new__(sdma.WeightTransport)
    transport.rt, transport.main = sdma._Runtime(), 1
    with pytest.raises(ValueError, match="descriptor failure"):
        with transport.capture_graph():
            raise ValueError("descriptor failure")
    assert not library.capturing
    assert sum(name == "aclmdlRIDestroy" for name, _ in library.calls) == 1
    with transport.capture_graph() as model:
        pass
    assert model.value == 9999
    assert not library.capturing
    assert sum(name == "aclmdlRIDestroy" for name, _ in library.calls) == 1


@pytest.mark.parametrize("count", [-1, 0, 5])
def test_invalid_chunk_rejected_before_capture(monkeypatch, count):
    library = RuntimeLibrary()
    monkeypatch.setattr(sdma.ct, "CDLL", lambda _: library)
    transport = sdma.WeightTransport.__new__(sdma.WeightTransport)
    transport.rt, transport.size, transport.width = sdma._Runtime(), 4, 4
    with pytest.raises(ValueError, match="must fit"):
        transport.capture([(100, 200, count)])
    assert library.calls == []
