# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-independent single output projection for head-bucket attention."""

import torch.nn.functional as F


def project_once(full_heads, weight, bias=None):
    """Apply W_o once after all head buckets are restored."""
    return F.linear(full_heads.reshape(full_heads.shape[0], -1), weight, bias)
