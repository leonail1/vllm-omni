# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend same-host weight prefetch with device-side chunk admission.

Ready notifications protect producer-to-reader visibility. Reader acknowledgements
protect the two source slots across chunks and graph replays. All transfers use
SDMA; the admission wait is a runtime task rather than a resident vector kernel.
"""

from __future__ import annotations

import ctypes as ct
import os
from contextlib import contextmanager
from math import prod

import torch
import torch.distributed as dist

_V, _U, _Q, _Z = ct.c_void_p, ct.c_uint32, ct.c_uint64, ct.c_size_t
_TRANSPORTS = {}


class _Runtime:
    def __init__(self):
        self.lib = ct.CDLL("libascendcl.so")
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
        return result.value


class WeightTransport:
    def __init__(self, cpu_group, chunk_bytes, *, slot_bytes=None):
        self.cpu_group = cpu_group
        self.rank, self.size = dist.get_rank(cpu_group), dist.get_world_size(cpu_group)
        if not 2 * 1024**2 <= chunk_bytes <= 4 * 1024**2:
            raise ValueError("SDMA weight prefetch requires a 2–4 MiB full-output chunk")
        self.width = slot_bytes or (chunk_bytes + self.size - 1) // self.size
        self.width = (self.width + 255) // 256 * 256
        self.device = torch.npu.current_device()
        self.rt = rt = _Runtime()
        self.plans = {}
        self.ranks = tuple(dist.get_process_group_ranks(cpu_group))
        self.attention = None
        self.storage = rt.handle("aclrtMalloc", 2 * self.width + 64, 0)
        self.gate = self.storage + 2 * self.width
        rt("aclrtMemset", self.gate, 64, 0, 64)
        pid = ct.c_int32()
        rt("aclrtDeviceGetBareTgid", ct.byref(pid))
        pids = self.exchange((pid.value, os.uname().nodename))
        if len({host for _, host in pids}) != 1:
            raise ValueError("SDMA weight prefetch requires all group members on one host")
        key = ct.create_string_buffer(65)
        rt("aclrtIpcMemGetExportKey", self.storage, 2 * self.width + 64, key, len(key), 0)
        self.key = key.value
        allowed = (ct.c_int32 * (self.size - 1))(*[p for i, (p, _) in enumerate(pids) if i != self.rank])
        rt("aclrtIpcMemSetImportPid", self.key, allowed, len(allowed))
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
        self.keys = self.exchange((self.key, notify_keys))
        self.peers, self.remote_notifies = [], []
        for rank, (memory_key, notification_keys) in enumerate(self.keys):
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
        self.go, self.start, self.producer_end, self.entry, self.exit, *self.done = [
            rt.handle("aclrtCreateEventExWithFlag", 1) for _ in range(self.size + 5)
        ]
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
        if key not in self.plans:
            chunks = []
            for item, (source, output) in zip(manifest.dtypes, buffers, strict=True):
                if not source.is_pinned():
                    raise ValueError("Captured SDMA prefetch requires pinned host weights")
                element_size = source.element_size()
                for chunk in item.chunks:
                    chunks.append(
                        (
                            source.data_ptr() + chunk.cpu_offset * element_size,
                            output.data_ptr() + chunk.full_offset * element_size,
                            chunk.local_numel * element_size,
                        )
                    )
            self.plans[key] = (self.capture(chunks), manifest, buffers)
        rt = self.rt
        # Accessing npu_stream drains PyTorch's host submission queue. Native
        # events then preserve the existing copy-stream / compute dependency.
        rt("aclrtRecordEvent", self.entry, copy_stream.npu_stream)
        rt("aclrtStreamWaitEvent", self.main, self.entry)
        rt("aclmdlRIExecuteAsync", self.plans[key][0], self.main)
        rt("aclrtRecordEvent", self.exit, self.main)
        rt("aclrtStreamWaitEvent", comm_stream.npu_stream, self.exit)

    def capture(self, chunks):
        rt, size = self.rt, self.size
        rt("aclmdlRICaptureBegin", self.main, 1)
        rt("aclrtRecordEvent", self.start, self.main)
        rt("aclrtStreamWaitEvent", self.producer, self.start)
        for index, (source, output, count) in enumerate(chunks):
            if count > self.width:
                raise ValueError("Weight chunk exceeds the shared source slot")
            slot = index % 2
            for peer in range(size):
                rt("aclrtWaitAndResetNotify", self.notifies[2 * size + slot * size + peer], self.producer, 0)
            rt("aclrtMemcpyAsync", self.storage + slot * self.width, count, source, count, 1, self.producer)
            for peer in range(size):
                rt("aclrtRecordNotify", self.remote_notifies[peer][slot * size + self.rank], self.producer)
            rt("aclrtValueWait", self.gate, 0, 1, self.main)
            rt("aclrtRecordEvent", self.go, self.main)
            for peer, stream in enumerate(self.readers):
                rt("aclrtStreamWaitEvent", stream, self.go)
                rt("aclrtWaitAndResetNotify", self.notifies[slot * size + peer], stream, 0)
                rt(
                    "aclrtMemcpyAsync",
                    output + peer * count,
                    count,
                    self.peers[peer] + slot * self.width,
                    count,
                    3,
                    stream,
                )
                rt("aclrtRecordNotify", self.remote_notifies[peer][2 * size + slot * size + self.rank], stream)
                rt("aclrtRecordEvent", self.done[peer], stream)
                rt("aclrtStreamWaitEvent", self.main, self.done[peer])
        rt("aclrtRecordEvent", self.producer_end, self.producer)
        rt("aclrtStreamWaitEvent", self.main, self.producer_end)
        model = _V()
        rt("aclmdlRICaptureEnd", self.main, ct.byref(model))
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
        for rank, (key, _) in enumerate(self.keys):
            if rank != self.rank:
                rt("aclrtIpcMemClose", key)
                for notify in self.remote_notifies[rank]:
                    if notify is not None:
                        rt("aclrtDestroyNotify", notify)
        dist.barrier(group=self.cpu_group)
        for notify in self.notifies:
            rt("aclrtDestroyNotify", notify)
        rt("aclrtIpcMemClose", self.key)
        for event in [self.go, self.start, self.producer_end, self.entry, self.exit, *self.done]:
            rt("aclrtDestroyEvent", event)
        for stream in [self.main, self.producer, *self.readers]:
            rt("aclrtDestroyStream", stream)
        rt("aclrtFree", self.storage)


def initialize_transport(device_group, cpu_group, chunk_bytes):
    if os.getenv("VLLM_OMNI_DLO_SDMA") == "1":
        if device_group in _TRANSPORTS:
            raise RuntimeError("A weight transport already owns this process group")
        _TRANSPORTS[device_group] = WeightTransport(cpu_group, chunk_bytes)


def get_transport(group):
    return _TRANSPORTS.get(group)


def close_transport(group):
    transport = _TRANSPORTS.get(group)
    if transport is not None:
        transport.close()
        del _TRANSPORTS[group]


@contextmanager
def attention_communication():
    """Admit attention once its inputs are ready, then resume weight chunks."""
    transports = list(_TRANSPORTS.values())
    if not transports:
        yield
        return
    stream = torch.npu.current_stream()
    for transport in transports:
        transport.rt("aclrtValueWrite", transport.gate, 1, 0, stream.npu_stream)
    try:
        yield
    finally:
        for transport in transports:
            transport.rt("aclrtValueWrite", transport.gate, 0, 0, stream.npu_stream)


def _attention_plan(source, size, rank, input_splits, output_splits, uniform_input, uniform_output):
    row_bytes = prod(source.shape[1:]) * source.element_size()
    if input_splits is None and output_splits is None:
        count = source.numel() * source.element_size() // size
        return [count] * size, [rank * count] * size, size * count
    if uniform_input:
        counts = [count * row_bytes for count in output_splits]
        return counts, [rank * count for count in counts], size * max(counts)
    if uniform_output:
        offset = sum(input_splits[:rank]) * row_bytes
        return [count * row_bytes for count in output_splits], [offset] * size, sum(input_splits) * row_bytes
    return None


def _direct_attention(channel, output, source, counts, offsets):
    rt, size = channel.rt, channel.size
    torch_stream = torch.npu.current_stream()
    stream = torch_stream.npu_stream
    source.record_stream(torch_stream)
    output.record_stream(torch_stream)
    for peer in range(size):
        rt("aclrtWaitAndResetNotify", channel.notifies[2 * size + peer], stream, 0)
    count = source.numel() * source.element_size()
    if count:
        rt("aclrtMemcpyAsync", channel.storage, count, source.data_ptr(), count, 3, stream)
    for peer in range(size):
        rt("aclrtRecordNotify", channel.remote_notifies[peer][channel.rank], stream)
    rt("aclrtRecordEvent", channel.go, stream)
    destination = output.data_ptr()
    for peer, reader in enumerate(channel.readers):
        rt("aclrtStreamWaitEvent", reader, channel.go)
        rt("aclrtWaitAndResetNotify", channel.notifies[peer], reader, 0)
        if counts[peer]:
            rt(
                "aclrtMemcpyAsync",
                destination,
                counts[peer],
                channel.peers[peer] + offsets[peer],
                counts[peer],
                3,
                reader,
            )
        rt("aclrtRecordNotify", channel.remote_notifies[peer][2 * size + channel.rank], reader)
        rt("aclrtRecordEvent", channel.done[peer], reader)
        rt("aclrtStreamWaitEvent", stream, channel.done[peer])
        destination += counts[peer]


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
        if owner is not None and input.device.type == "npu" and output.device == input.device:
            plan = _attention_plan(
                input, owner.size, owner.rank, input_split_sizes, output_split_sizes, uniform_input, uniform_output
            )
            # The supported Ulysses layouts give every rank the same global
            # capacity check, including variable and empty sequence shards.
            if plan is not None and plan[2] <= 32 * 1024**2:
                if owner.attention is None:
                    owner.attention = WeightTransport(owner.cpu_group, 4 * 1024**2, slot_bytes=16 * 1024**2)
                channel = owner.attention
    if channel is not None:
        source = input.contiguous()
        if torch_npu.get_npu_format(source) not in (0, 2):
            source = torch_npu.npu_format_cast(source, 2)
        destination = output
        if not output.is_contiguous() or torch_npu.get_npu_format(output) not in (0, 2):
            destination = torch.empty(output.shape, dtype=output.dtype, device=output.device)
    with attention_communication():
        if channel is not None:
            _direct_attention(channel, destination, source, plan[0], plan[1])
        else:
            return dist.all_to_all_single(
                output, input, output_split_sizes=output_split_sizes, input_split_sizes=input_split_sizes, group=group
            )
    if destination is not output:
        output.copy_(destination)
    return None
