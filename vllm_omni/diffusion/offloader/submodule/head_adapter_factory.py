# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One shared native alltoall communicator/stream across sequential Blocks."""

import torch

from .common.head_workspace_cache import HeadWorkspaceCache


class HeadAdapterFactory:
    def __init__(self, group, buckets=2):
        if buckets < 1:
            raise ValueError("Need at least one head bucket")
        self.group, self.buckets = group, buckets
        self.stream = torch.npu.Stream()
        self.workspaces = {}

    def __call__(self, block):
        name = type(block).__name__
        if name == "MiniMaxH3TokenRefinerBlock":
            # Replicated text refinement precedes sequence sharding.
            return []
        if name == "MiniMaxH3DiTBlock":
            from .models.minimax_h3.h3_bucket_adapter import H3BucketAdapter

            modules, adapter = [block.attn], H3BucketAdapter
        else:
            raise ValueError(f"No verified head adapter for {name}")
        instances = [
            adapter(
                module,
                self.group,
                self.buckets,
                stream=self.stream,
            )
            for module in modules
        ]
        for instance in instances:
            # Blocks run sequentially. Every input collective waits on a main
            # stream event after the preceding Block consumed its outputs.
            # Thus equal-layout adapters can reuse receive buffers safely.
            key = (adapter, instance.plan)
            if key not in self.workspaces:
                self.workspaces[key] = HeadWorkspaceCache()
            instance.workspace = self.workspaces[key]
        return instances
