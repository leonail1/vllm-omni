# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pack dense host weights once; device forwards use only narrow views.

Only enable this transform together with the matching bucketed forward.
O weights are stored as contiguous [input, output] rows so each bucket
can feed GEMM through a transpose view without gathering strided columns.
Quantized/TensorParallel methods need own adapters.
"""

from dataclasses import dataclass, field

import torch

from .head_buckets import HeadBucketPlan


@dataclass(frozen=True)
class BucketWeightLayout:
    plan: HeadBucketPlan
    qkv_spans: tuple[tuple[int, int], ...] = field(init=False)
    output_spans: tuple[tuple[int, int], ...] = field(init=False)

    def __post_init__(self):
        qkv_spans, output_spans = [], []
        qkv_offset = output_offset = 0
        for bucket in range(self.plan.buckets):
            kv_heads = sum(map(len, self.plan.head_ranges(bucket, kv=True)))
            q_heads = kv_heads * self.plan.query_heads // self.plan.kv_heads
            qkv_count = (q_heads + 2 * kv_heads) * self.plan.head_dim
            output_count = q_heads * self.plan.head_dim
            qkv_spans.append((qkv_offset, qkv_count))
            output_spans.append((output_offset, output_count))
            qkv_offset += qkv_count
            output_offset += output_count
        object.__setattr__(self, "qkv_spans", tuple(qkv_spans))
        object.__setattr__(self, "output_spans", tuple(output_spans))

    def _validate_host(self, tensor):
        if tensor.device.type != "cpu" or not tensor.is_floating_point():
            raise ValueError("Pack dense CPU floating-point weights before device transfer")

    def pack_qkv(self, tensor):
        self._validate_host(tensor)
        expected = (self.plan.query_heads + 2 * self.plan.kv_heads) * self.plan.head_dim
        if tensor.ndim not in (1, 2) or tensor.shape[0] != expected:
            raise ValueError("Expected [all Q rows; all K rows; all V rows]")
        rows = [i for b in range(self.plan.buckets) for i in self.plan.qkv_indices(b)]
        return tensor.index_select(0, torch.tensor(rows))

    def pack_output(self, weight):
        self._validate_host(weight)
        if weight.ndim != 2 or weight.shape[1] != self.plan.query_heads * self.plan.head_dim:
            raise ValueError("O projection input channels do not match query heads")
        columns = [i for b in range(self.plan.buckets) for i in self.plan.channel_indices(b)]
        return weight.index_select(1, torch.tensor(columns)).t().contiguous()

    def unpack_qkv(self, tensor):
        """Restore the original row order after DLO restores packed CPU weights."""
        self._validate_host(tensor)
        rows = [i for b in range(self.plan.buckets) for i in self.plan.qkv_indices(b)]
        if tensor.ndim not in (1, 2) or tensor.shape[0] != len(rows):
            raise ValueError("Expected materialized bucket-packed QKV rows")
        restored = torch.empty_like(tensor)
        restored.index_copy_(0, torch.tensor(rows), tensor)
        return restored

    def qkv_view(self, packed, bucket):
        if not 0 <= bucket < len(self.qkv_spans):
            raise IndexError(bucket)
        return packed.narrow(0, *self.qkv_spans[bucket])

    def output_view(self, packed, bucket):
        if not 0 <= bucket < len(self.output_spans):
            raise IndexError(bucket)
        # F.linear transposes this view back to the contiguous stored bucket.
        return packed.narrow(0, *self.output_spans[bucket]).t()
