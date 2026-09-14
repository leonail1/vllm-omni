# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-independent QKV projection helpers for head-bucket adapters."""

import torch.nn.functional as F


def project_qkv(x, weight, bias, *, world_size, heads, head_dim):
    """Project one head bucket and return contiguous Q/K/V TND-ready tensors."""
    projected = F.linear(x, weight, bias)
    q, k, v = projected.chunk(3, dim=-1)
    shape = (x.shape[0], world_size * heads, head_dim)
    return tuple(t.reshape(shape) for t in (q, k, v)), projected


def pack_qkv_for_alltoall(qkv, world_size, heads, head_dim):
    return tuple(t.reshape(t.shape[0], world_size, heads, head_dim).permute(1, 0, 2, 3).contiguous() for t in qkv)
