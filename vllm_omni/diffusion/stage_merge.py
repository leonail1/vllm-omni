# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for merging diffusion stage outputs."""

from __future__ import annotations

from typing import Any

import torch


def merge_latent_outputs(values: list[Any]) -> Any | None:
    """Merge per-prompt latent outputs without dropping later prompts."""

    non_null = [value for value in values if value is not None]
    if not non_null:
        return None
    if len(non_null) == 1:
        return non_null[0]

    # Latent outputs from multi-prompt batches are batch-major tensors.  When
    # their row suffixes match, concatenate instead of keeping only one prompt.
    if all(torch.is_tensor(value) for value in non_null):
        first = non_null[0]
        if all(
            value.device == first.device
            and value.dtype == first.dtype
            and value.ndim == first.ndim
            and tuple(value.shape[1:]) == tuple(first.shape[1:])
            for value in non_null
        ):
            return torch.cat(non_null, dim=0)

    # Fall back to a list for heterogeneous payloads such as per-prompt custom
    # latent containers.  This keeps every output visible to the caller.
    merged: list[Any] = []
    for value in non_null:
        if isinstance(value, (list, tuple)):
            merged.extend(value)
        else:
            merged.append(value)
    return merged
