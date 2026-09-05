# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend same-host weight prefetch with device-side chunk admission.

Ready notifications protect producer-to-reader visibility. Reader acknowledgements
protect two source windows across graph replays. Each window batches four
producer notifications while admission remains one full-output chunk. All transfers use
SDMA; the admission wait is a runtime task rather than a resident vector kernel.
"""

from __future__ import annotations

import ctypes as ct
import os
from contextlib import ExitStack, contextmanager
from math import prod
from pathlib import Path

import torch
import torch.distributed as dist

_V, _U, _Q, _Z = ct.c_void_p, ct.c_uint32, ct.c_uint64, ct.c_size_t
_TRANSPORTS = {}


class _Runtime:
    def __init__(self):
        self.lib = ct.CDLL("libascendcl.so")
        self.resources, self.imports = ExitStack(), ExitStack()
        signatures = {
            "aclrtMalloc": [ct.POINTER(_V), _Z, ct.c_int],
            "aclrtFree": [_V],
            "aclrtMemset": [_V, _Z, ct.c_int, _Z],
            "aclrtDeviceGetBareTgid": [ct.POINTER(ct.c_int32)],
            "aclrtIpcMemGetExportKey": [_V, _Z, ct.c_char_p, _Z, _Q],
            "aclrtIpcMemSetImportPid": [ct.c_char_p, ct.POINTER(ct.c_int32), _Z],
            "aclrtIpcMemImportByKey": [ct.POINTER(_V), ct.c_char_p, _Q],
            "aclrtIpcMemClose": [ct.c_char_p],
            "aclrtCreateNotify": [ct.POINTER(_V), _Q],
            "aclrtDestroyNotify": [_V],
            "aclrtNotifyGetExportKey": [_V, ct.c_char_p, _Z, _Q],
            "aclrtNotifySetImportPid": [_V, ct.POINTER(ct.c_int32), _Z],
            "aclrtNotifyImportByKey": [ct.POINTER(_V), ct.c_char_p, _Q],
            "aclrtRecordNotify": [_V, _V],
            "aclrtWaitAndResetNotify": [_V, _V, _U],
            "aclrtCreateStreamWithConfig": [ct.POINTER(_V), _U, _U],
            "aclrtSynchronizeStream": [_V],
            "aclrtDestroyStream": [_V],
            "aclrtCreateEventExWithFlag": [ct.POINTER(_V), _U],
            "aclrtRecordEvent": [_V, _V],
            "aclrtStreamWaitEvent": [_V, _V],
            "aclrtDestroyEvent": [_V],
            "aclrtMemcpyAsync": [_V, _Z, _V, _Z, ct.c_int, _V],
            "aclrtValueWrite": [_V, _Q, _U, _V],
            "aclrtValueWait": [_V, _Q, _U, _V],
            "aclmdlRICaptureBegin": [_V, ct.c_int],
            "aclmdlRICaptureEnd": [_V, ct.POINTER(_V)],
            "aclmdlRIExecuteAsync": [_V, _V],
            "aclmdlRIDestroy": [_V],
        }
        for name, args in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, ct.c_int

    def __call__(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            raise RuntimeError(f"{name} returned {code}")

    def handle(self, name, *args):
        result = _V()
        self(name, ct.byref(result), *args)
        if name == "aclrtIpcMemImportByKey":
            self.imports.callback(self, "aclrtIpcMemClose", args[0])
        else:
            release = {
                "aclrtMalloc": "aclrtFree",
                "aclrtCreateNotify": "aclrtDestroyNotify",
                "aclrtNotifyImportByKey": "aclrtDestroyNotify",
                "aclrtCreateStreamWithConfig": "aclrtDestroyStream",
                "aclrtCreateEventExWithFlag": "aclrtDestroyEvent",
            }[name]
            resources = self.imports if name == "aclrtNotifyImportByKey" else self.resources
            resources.callback(self, release, result.value)
        return result.value

    def copy(self, destination, source, count, stream, kind=3):
        self("aclrtMemcpyAsync", destination, count, source, count, kind, stream)


class WeightTransport:
    def __init__(self, cpu_group, chunk_bytes, *, slot_bytes=None):
        self.cpu_group = cpu_group
        self.rank, self.size = dist.get_rank(cpu_group), dist.get_world_size(cpu_group)
        if not 2 * 1024**2 <= chunk_bytes <= 4 * 1024**2:
            raise ValueError("SDMA weight prefetch requires a 2–4 MiB full-output chunk")
        self.width = slot_bytes or (chunk_bytes + self.size - 1) // self.size
        self.width = (self.width + 255) // 256 * 256
        self.window_chunks = 4 if slot_bytes is None else 1
        self.slot_width = self.width * self.window_chunks
        self.device = torch.npu.current_device()
        self.rt = rt = _Runtime()
        self.plans = {}
        self.ranks = tuple(dist.get_process_group_ranks(cpu_group))
        self.attention = None
        self.attention_stream = None
        self.prefetch_stream = None
        self.result = None
        try:
            self._open()
        except BaseException:
            try:
                rt.imports.close()
            finally:
                rt.resources.close()
            raise

    def _open(self):
        rt = self.rt
        pid = ct.c_int32()
        rt("aclrtDeviceGetBareTgid", ct.byref(pid))
        pids = self.exchange((pid.value, os.uname().nodename))
        if len({host for _, host in pids}) != 1:
            raise ValueError("SDMA weight prefetch requires all group members on one host")
        self.storage = rt.handle("aclrtMalloc", 2 * self.slot_width + 64, 0)
        self.gate = self.storage + 2 * self.slot_width
        rt("aclrtMemset", self.gate, 64, 0, 64)
        key = ct.create_string_buffer(65)
        rt("aclrtIpcMemGetExportKey", self.storage, 2 * self.slot_width + 64, key, len(key), 0)
        memory_key = key.value
        rt.resources.callback(rt, "aclrtIpcMemClose", memory_key)
        allowed = (ct.c_int32 * (self.size - 1))(*[p for i, (p, _) in enumerate(pids) if i != self.rank])
        rt("aclrtIpcMemSetImportPid", memory_key, allowed, len(allowed))
        self.notifies = [rt.handle("aclrtCreateNotify", 0) for _ in range(4 * self.size)]
        notify_keys = []
        for index, notify in enumerate(self.notifies):
            sender = index % self.size
            value = None
            if sender != self.rank:
                rt("aclrtNotifyGetExportKey", notify, key, len(key), 0)
                value = key.value
                sender_pid = ct.c_int32(pids[sender][0])
                rt("aclrtNotifySetImportPid", notify, ct.byref(sender_pid), 1)
            notify_keys.append(value)
        keys = self.exchange((memory_key, notify_keys))
        self.peers, self.remote_notifies = [], []
        for rank, (memory_key, notification_keys) in enumerate(keys):
            self.peers.append(self.storage if rank == self.rank else rt.handle("aclrtIpcMemImportByKey", memory_key, 1))
            self.remote_notifies.append(
                self.notifies
                if rank == self.rank
                else [
                    rt.handle("aclrtNotifyImportByKey", key, 2) if index % self.size == self.rank else None
                    for index, key in enumerate(notification_keys)
                ]
            )
        self.main, self.producer, *self.readers = [
            rt.handle("aclrtCreateStreamWithConfig", 0, 1) for _ in range(self.size + 2)
        ]
        self.go, self.start, self.producer_end, *self.done = [
            rt.handle("aclrtCreateEventExWithFlag", 1) for _ in range(self.size + 3)
        ]
        self.chunk_notifies = [rt.handle("aclrtCreateNotify", 0) for _ in range(2 * self.size)]
        for notify in self.notifies[2 * self.size :]:
            rt("aclrtRecordNotify", notify, self.producer)
        rt("aclrtSynchronizeStream", self.producer)

    def exchange(self, value):
        values = [None] * self.size
        dist.all_gather_object(values, value, group=self.cpu_group)
        return values

    def prefetch(self, manifest, cpu_shards, outputs, copy_stream, comm_stream):
        buffers = tuple((cpu_shards[item.dtype], outputs[item.dtype]) for item in manifest.dtypes)
        key = (id(manifest), tuple((source.data_ptr(), output.data_ptr()) for source, output in buffers))
        if buffers and key not in self.plans:
            # Capture only after previous host submissions have reached the runtime.
            torch.npu.current_stream().npu_stream
            chunks = []
            for item, (source, output) in zip(manifest.dtypes, buffers, strict=True):
                if not source.is_pinned():
                    raise ValueError("Captured SDMA prefetch requires pinned host weights")
                if source.dtype != output.dtype or not source.is_contiguous() or not output.is_contiguous():
                    raise ValueError("SDMA prefetch requires contiguous matching-dtype buffers")
                if output.device.type != "npu" or output.device.index != self.device:
                    raise ValueError("Weight output must belong to the transport device")
                element_size = source.element_size()
                for chunk in item.chunks:
                    if not (
                        0 <= chunk.cpu_offset <= source.numel() - chunk.local_numel
                        and 0 <= chunk.full_offset <= output.numel() - self.size * chunk.local_numel
                    ):
                        raise ValueError("Weight chunk exceeds its source or output buffer")
                    chunks.append(
                        (
                            source.data_ptr() + chunk.cpu_offset * element_size,
                            output.data_ptr() + chunk.full_offset * element_size,
                            chunk.local_numel * element_size,
                        )
                    )
            self.plans[key] = (self.capture(chunks), manifest, buffers)
        comm_stream.wait_stream(copy_stream)
        if self.prefetch_stream is not None and self.prefetch_stream != comm_stream:
            comm_stream.wait_stream(self.prefetch_stream)
        self.prefetch_stream = comm_stream
        if buffers:
            with torch.npu.stream(comm_stream):
                torch.ops.vllm_omni_sdma.replay(None, [output for _, output in buffers], self.plans[key][0], 0, 0, [])

    @contextmanager
    def capture_graph(self):
        model = _V()
        self.rt("aclmdlRICaptureBegin", self.main, 1)
        try:
            yield model
            self.rt("aclmdlRICaptureEnd", self.main, ct.byref(model))
        except BaseException:
            if not model.value:
                self.rt.lib.aclmdlRICaptureEnd(self.main, ct.byref(model))
            if model.value:
                self.rt.lib.aclmdlRIDestroy(model)
            raise

    def capture(self, chunks):
        rt, size = self.rt, self.size
        if any(count <= 0 or count > self.width for _, _, count in chunks):
            raise ValueError("Weight chunk must fit its portion of the shared window")
        with self.capture_graph() as model:
            rt("aclrtRecordEvent", self.start, self.main)
            rt("aclrtStreamWaitEvent", self.producer, self.start)
            for reader in self.readers:
                rt("aclrtStreamWaitEvent", reader, self.start)
            for index, begin in enumerate(range(0, len(chunks), self.window_chunks)):
                batch = chunks[begin : begin + self.window_chunks]
                slot = index % 2
                base = slot * self.slot_width
                for peer in range(size):
                    rt("aclrtWaitAndResetNotify", self.notifies[2 * size + slot * size + peer], self.producer, 0)
                offset = 0
                for source, output, count in batch:
                    rt.copy(self.storage + base + offset, source, count, self.producer, kind=1)
                    offset += count
                for peer in range(size):
                    rt("aclrtRecordNotify", self.remote_notifies[peer][slot * size + self.rank], self.producer)
                for peer, reader in enumerate(self.readers):
                    rt("aclrtWaitAndResetNotify", self.notifies[slot * size + peer], reader, 0)
                offset = 0
                for source, output, count in batch:
                    # Each full-output chunk retains its admission and completion barrier.
                    rt("aclrtValueWait", self.gate, 0, 1, self.main)
                    for peer in range(size):
                        rt("aclrtRecordNotify", self.chunk_notifies[peer], self.main)
                    for peer, reader in enumerate(self.readers):
                        rt("aclrtWaitAndResetNotify", self.chunk_notifies[peer], reader, 0)
                        rt.copy(output + peer * count, self.peers[peer] + base + offset, count, reader)
                        rt("aclrtRecordNotify", self.chunk_notifies[size + peer], reader)
                        rt("aclrtWaitAndResetNotify", self.chunk_notifies[size + peer], self.main, 0)
                    offset += count
                for peer, reader in enumerate(self.readers):
                    rt("aclrtRecordNotify", self.remote_notifies[peer][2 * size + slot * size + self.rank], reader)
            for peer, reader in enumerate(self.readers):
                rt("aclrtRecordEvent", self.done[peer], reader)
                rt("aclrtStreamWaitEvent", self.main, self.done[peer])
            rt("aclrtRecordEvent", self.producer_end, self.producer)
            rt("aclrtStreamWaitEvent", self.main, self.producer_end)
        return model.value

    def close(self):
        if self.attention is not None:
            self.attention.close()
            self.attention = None
        torch.npu.synchronize()
        rt = self.rt
        for model, _, _ in self.plans.values():
            rt("aclmdlRIDestroy", model)
        self.plans.clear()
        dist.barrier(group=self.cpu_group)
        rt.imports.close()
        dist.barrier(group=self.cpu_group)
        rt.resources.close()
        self.result = None


def initialize_transport(device_group, cpu_group, chunk_bytes):
    if os.getenv("VLLM_OMNI_DLO_SDMA") == "1":
        if device_group in _TRANSPORTS:
            raise RuntimeError("A weight transport already owns this process group")
        import torch_npu
        from torch.utils.cpp_extension import load

        # Build once in PyTorch's shared extension cache; its file lock serializes
        # worker startup. Only the explicitly enabled transport needs a compiler.
        npu = Path(torch_npu.__file__).parent
        cann = Path(os.environ["ASCEND_HOME_PATH"])
        load(
            name="vllm_omni_sdma",
            sources=[str(Path(__file__).with_suffix(".cpp"))],
            extra_include_paths=[str(npu / "include"), str(cann / "include")],
            extra_cflags=["-O2"],
            extra_ldflags=[
                f"-L{npu / 'lib'}",
                f"-Wl,-rpath,{npu / 'lib'}",
                "-ltorch_npu",
                f"-L{cann / 'lib64'}",
                f"-Wl,-rpath,{cann / 'lib64'}",
                "-lascendcl",
            ],
            with_cuda=False,
            is_python_module=False,
        )
        _TRANSPORTS[device_group] = WeightTransport(cpu_group, chunk_bytes)


def get_transport(group):
    return _TRANSPORTS.get(group)


def close_transport(group):
    transport = _TRANSPORTS.get(group)
    if transport is not None:
        transport.close()
        del _TRANSPORTS[group]


def _attention_transports():
    transports = [value for value in _TRANSPORTS.values() if value.device == torch.npu.current_device()]
    stream = torch.npu.current_stream()
    for transport in transports:
        previous = transport.attention_stream
        if previous is not None and previous != stream:
            transport.rt("aclrtRecordEvent", transport.go, previous.npu_stream)
            transport.rt("aclrtStreamWaitEvent", stream.npu_stream, transport.go)
        transport.attention_stream = stream
    return transports


@contextmanager
def attention_communication():
    """Admit attention once its inputs are ready, then resume weight chunks."""
    transports = _attention_transports() if _TRANSPORTS else []
    for transport in transports:
        transport.rt("aclrtValueWrite", transport.gate, 1, 0, torch.npu.current_stream().npu_stream)
    try:
        yield
    finally:
        for transport in transports:
            transport.rt("aclrtValueWrite", transport.gate, 0, 0, torch.npu.current_stream().npu_stream)


def _attention_plan(source, size, rank, input_splits, output_splits, uniform_input, uniform_output):
    row_bytes = prod(source.shape[1:]) * source.element_size()
    if input_splits is None and output_splits is None:
        if not source.ndim or source.shape[0] % size:
            raise ValueError("Equal all-to-all requires a divisible leading dimension")
        count = source.numel() * source.element_size() // size
        return [count] * size, [rank * count] * size, size * count
    if uniform_input:
        counts = [count * row_bytes for count in output_splits]
        return counts, [rank * count for count in counts], size * max(counts)
    if uniform_output:
        offset = sum(input_splits[:rank]) * row_bytes
        return [count * row_bytes for count in output_splits], [offset] * size, size * max(input_splits) * row_bytes
    return None


def _direct_attention(channel, output, source, counts, offsets):
    rt, size = channel.rt, channel.size
    key = (tuple(counts), tuple(offsets))
    if key not in channel.plans:
        # Finish queued host submissions before capturing a new native graph.
        torch.npu.current_stream().npu_stream
        main = channel.main
        with channel.capture_graph() as model:
            for peer in range(size):
                rt("aclrtRecordNotify", channel.remote_notifies[peer][channel.rank], main)
            rt("aclrtRecordEvent", channel.go, main)
            destination = channel.result.data_ptr()
            for peer, reader in enumerate(channel.readers):
                rt("aclrtStreamWaitEvent", reader, channel.go)
                rt("aclrtWaitAndResetNotify", channel.notifies[peer], reader, 0)
                if counts[peer]:
                    rt.copy(destination, channel.peers[peer] + offsets[peer], counts[peer], reader)
                rt("aclrtRecordNotify", channel.remote_notifies[peer][2 * size + channel.rank], reader)
                rt("aclrtRecordEvent", channel.done[peer], reader)
                rt("aclrtStreamWaitEvent", main, channel.done[peer])
                destination += counts[peer]
            # A completed replay releases our source pool on every peer. This
            # protects the next invocation's staging copy on the caller stream.
            for peer in range(size):
                rt("aclrtWaitAndResetNotify", channel.notifies[2 * size + peer], main, 0)
        channel.plans[key] = (model.value, None, ())
    gates = [transport.gate for transport in _attention_transports()]
    torch.ops.vllm_omni_sdma.replay(
        source, [output], channel.plans[key][0], channel.storage, channel.result.data_ptr(), gates
    )


def all_to_all_single(
    output,
    input,
    output_split_sizes=None,
    input_split_sizes=None,
    group=None,
    *,
    uniform_input=False,
    uniform_output=False,
):
    channel = plan = None
    if _TRANSPORTS:
        import torch_npu

        ranks = tuple(dist.get_process_group_ranks(group or dist.group.WORLD))
        owner = next((value for value in _TRANSPORTS.values() if value.ranks == ranks), None)
        if (
            owner is not None
            and input.device.type == "npu"
            and output.device == input.device
            and input.device.index == owner.device
        ):
            plan = _attention_plan(
                input, owner.size, owner.rank, input_split_sizes, output_split_sizes, uniform_input, uniform_output
            )
            # The supported Ulysses layouts give every rank the same global
            # capacity check, including variable and empty sequence shards.
            if plan is not None and plan[2] <= 32 * 1024**2:
                if (
                    input.dtype != output.dtype
                    or sum(plan[0]) != output.numel() * output.element_size()
                    or input.numel() * input.element_size() > plan[2]
                ):
                    raise ValueError("All-to-all buffers do not match the split layout")
                if owner.attention is None:
                    owner.attention = WeightTransport(owner.cpu_group, 4 * 1024**2, slot_bytes=16 * 1024**2)
                    owner.attention.result = torch.empty(32 * 1024**2, dtype=torch.uint8, device=input.device)
                    child = owner.attention
                    for notify in child.notifies[2 * child.size : 3 * child.size]:
                        child.rt("aclrtWaitAndResetNotify", notify, child.producer, 0)
                    child.rt("aclrtSynchronizeStream", child.producer)
                channel = owner.attention
    if channel is not None:
        source = input.contiguous()
        if torch_npu.get_npu_format(source) not in (0, 2):
            source = torch_npu.npu_format_cast(source, 2)
        destination = output
        if not output.is_contiguous() or torch_npu.get_npu_format(output) not in (0, 2):
            destination = torch.empty(output.shape, dtype=output.dtype, device=output.device)
        _direct_attention(channel, destination, source, plan[0], plan[1])
        if destination is not output:
            output.copy_(destination)
        return None
    with attention_communication():
        return dist.all_to_all_single(
            output, input, output_split_sizes=output_split_sizes, input_split_sizes=input_split_sizes, group=group
        )
