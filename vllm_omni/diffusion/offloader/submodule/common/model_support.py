# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Limit head splitting to the model adapters shipped in this change."""


def supports_block_group(blocks):
    """Auxiliary language/encoder stacks keep their existing block transport."""
    supported = {
        "MiniMaxH3DiTBlock",
        "MiniMaxH3TokenRefinerBlock",
    }
    return bool(blocks) and all(type(block).__name__ in supported for block in blocks)
