# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared tensor utilities for distributed layerwise offload.

These helpers are used by both DistributedLayerwiseOffloadHook and
DistributedLayerwiseOffloadBackend, and can be reused by other
offload backends.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.distributed.tensor import DTensor


def dtype_size(dtype: torch.dtype) -> int:
    """Return element size in bytes for a torch.dtype."""
    return torch.empty(1, dtype=dtype).element_size()


def is_dtensor(t: torch.Tensor) -> bool:
    """Check if tensor is a DTensor."""
    return isinstance(t, DTensor)


def set_tensor_storage(target: torch.Tensor, value: torch.Tensor) -> None:
    """Replace target's underlying storage with value (zero-copy)."""
    if is_dtensor(target):
        target._local_tensor = value
    else:
        target.data = value


# Placeholders are immutable zero-element/meta tensors; caching them keeps the
# retire path free of repeated allocation calls (design section 26.3: no
# Tensor allocation on the hot path).
_PLACEHOLDER_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}


def make_offload_placeholder(tensor: torch.Tensor) -> torch.Tensor:
    """Return a zero-element placeholder to free GPU memory (cached)."""
    if is_dtensor(tensor):
        key: tuple[Any, ...] = ("meta", tensor.dtype, tuple(tensor.to_local().shape))
    else:
        key = (str(tensor.device), tensor.dtype)
    placeholder = _PLACEHOLDER_CACHE.get(key)
    if placeholder is None:
        if is_dtensor(tensor):
            placeholder = torch.empty(key[2], device="meta", dtype=tensor.dtype)
        else:
            placeholder = torch.empty((0,), device=tensor.device, dtype=tensor.dtype)
        _PLACEHOLDER_CACHE[key] = placeholder
    return placeholder


def is_materialized_tensor(t: torch.Tensor) -> bool:
    """Check if tensor holds real data (not meta or empty placeholder)."""
    if is_dtensor(t):
        local_t = t.to_local()
        return not local_t.is_meta
    return not t.is_meta and t.data.numel() > 0
