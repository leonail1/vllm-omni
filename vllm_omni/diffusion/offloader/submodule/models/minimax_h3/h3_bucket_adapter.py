# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 dense single-request Ulysses head pipeline."""

from types import MethodType

import torch
import torch.distributed as dist
import torch_npu

from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope

from ...common.bucket_weight_layout import BucketWeightLayout
from ...common.head_buckets import HeadBucketPlan
from ...common.head_output_layout import assemble_output_bucket
from ...common.head_workspace_cache import HeadWorkspaceCache
from ...common.output_projection import project_once
from ...common.qkv import pack_qkv_for_alltoall, project_qkv


class H3BucketAdapter:
    def __init__(self, module, group, buckets=2, stream=None):
        if getattr(module, "to_gate_compress", None) is not None:
            raise ValueError("H3 head-split supports dense attention, not FastH3 VSA")
        if getattr(module, "_h3_bucket_adapter", None) is not None:
            raise ValueError("H3 head-split adapter is already installed")
        if module.num_heads != module.num_kv_heads or module.qkv_proj.total_num_heads != module.num_heads:
            raise ValueError("H3 adapter requires dense MHA and TP=1")
        self.module, self.group = module, group
        self.plan = HeadBucketPlan(
            module.num_heads,
            module.num_kv_heads,
            module.head_dim,
            dist.get_world_size(group),
            buckets,
        )
        self.layout = BucketWeightLayout(self.plan)
        for linear in (module.qkv_proj, module.out_proj):
            if linear.weight.device.type != "cpu" or type(linear.quant_method).__name__ != "UnquantizedLinearMethod":
                raise ValueError("Install on unquantized CPU weights")
        # Prepare everything before changing the module, so allocation failure
        # cannot leave reordered weights behind with the original forward.
        packed_weight = self.layout.pack_qkv(module.qkv_proj.weight.detach())
        packed_bias = None if module.qkv_proj.bias is None else self.layout.pack_qkv(module.qkv_proj.bias.detach())
        self.stream = torch.npu.Stream() if stream is None else stream
        self.input_ready = [torch.npu.Event() for _ in range(buckets)]
        self.output_ready = [torch.npu.Event() for _ in range(buckets)]
        self.done, self.used = torch.npu.Event(), False
        self.workspace = HeadWorkspaceCache()
        self._original_forward = module.forward
        self._had_instance_forward = "forward" in module.__dict__
        self._closed = False
        module.qkv_proj.weight.data = packed_weight
        if packed_bias is not None:
            module.qkv_proj.bias.data = packed_bias
        module.forward = MethodType(lambda m, x, **kwargs: self.forward(x, **kwargs), module)
        module._h3_bucket_adapter = self

    def forward(
        self,
        x,
        *,
        rope_table,
        cu_seqlens,
        max_seqlen,
        packed_total=None,
        num_requests=1,
        sp_seq_lens=None,
        video_layout=None,
        vsa_prefix_segments=(),
    ):
        if self._closed:
            raise RuntimeError("H3 head-split adapter has been closed")
        if getattr(self.module, "to_gate_compress", None) is not None:
            raise ValueError("H3 head-split supports dense attention, not FastH3 VSA")
        # Dense attention ignores VSA partition metadata just like upstream.
        m, plan, layout = self.module, self.plan, self.layout
        world, tokens, d = plan.world_size, x.shape[0], plan.head_dim
        if num_requests != 1 or packed_total != world * tokens:
            raise ValueError("Only equal SP shards of one packed request are supported")
        if not 0 < max_seqlen <= packed_total:
            raise ValueError("Invalid packed valid length")
        if sp_seq_lens is not None and any(length != tokens for length in sp_seq_lens):
            raise ValueError("Uneven sequence shards need a separate collective plan")
        main, comm = torch.npu.current_stream(), self.stream
        if self.used:
            main.wait_event(self.done)
            comm.wait_event(self.done)
        key = (tokens, x.dtype, x.device)

        def allocate():
            buckets = [
                (
                    tuple(
                        torch.empty(world, tokens, len(plan.head_ranges(b)[0]), d, device=x.device, dtype=x.dtype)
                        for _ in range(3)
                    ),
                    torch.empty(
                        world,
                        tokens,
                        len(plan.head_ranges(b)[0]),
                        d,
                        device=x.device,
                        dtype=x.dtype,
                    ),
                )
                for b in range(plan.buckets)
            ]
            return buckets, torch.empty(tokens, plan.query_heads, d, device=x.device, dtype=x.dtype)

        with (
            self.workspace.lease(key, allocate, stream=main, stream_key=(main.device, main.npu_stream)) as buffers,
        ):
            buffers, full_output = buffers
            inputs = []
            outputs: list[tuple[torch.Tensor, dist.Work]] = []
            keepalive: list[torch.Tensor] = []

            def launch_input(bucket):
                h = len(plan.head_ranges(bucket)[0])
                bias = None if m.qkv_proj.bias is None else layout.qkv_view(m.qkv_proj.bias, bucket)
                (q, k, v), projected = project_qkv(
                    x,
                    layout.qkv_view(m.qkv_proj.weight, bucket),
                    bias,
                    world_size=world,
                    heads=h,
                    head_dim=d,
                )
                if rope_table is None:
                    q, k = m.q_norm(q), m.k_norm(k)
                else:
                    q, k = fused_qk_norm_rope(
                        q,
                        k,
                        m.q_norm.weight,
                        m.k_norm.weight,
                        rope_table,
                        m.q_norm.variance_epsilon,
                    )
                # Separate-QKV is the only supported path: each Q/K/V tensor
                # receives directly into contiguous TND storage.
                sends = pack_qkv_for_alltoall((q, k, v), world, h, d)
                recv = buffers[bucket][0]
                recvs = recv
                self.input_ready[bucket].record(main)
                with torch.npu.stream(comm):
                    comm.wait_event(self.input_ready[bucket])
                    # All ranks enqueue Q, K, V in identical order. Do not wait here:
                    # the main stream can project the next bucket concurrently.
                    works = tuple(
                        dist.all_to_all_single(dst, src, group=self.group, async_op=True)
                        for dst, src in zip(recvs, sends)
                    )
                inputs.append((recv, works, h))
                keepalive.extend((projected, q, k, v, *sends))

            def contribute(bucket, total):
                recv, work = outputs[bucket]
                work.wait()
                # Restore each bucket directly into the final token/head layout;
                # output projection runs exactly once after all buckets arrive.
                local_heads = plan.head_ranges(bucket)[0]
                assemble_output_bucket(full_output, recv, local_heads.start)
                return None

            total = None
            launch_input(0)
            for bucket in range(plan.buckets):
                if bucket + 1 < plan.buckets:
                    launch_input(bucket + 1)
                recv, works, h = inputs[bucket]
                for work in works:
                    work.wait()
                # Preserve the original H3 packed-varlen contract per bucket.
                # The suffix padding is its own document, never a KV prefix
                # attended by padded query rows. No host/device scalar sync.
                if cu_seqlens.ndim != 1 or cu_seqlens.numel() != (2 if max_seqlen == packed_total else 3):
                    raise ValueError("H3 TND requires the original single-request real/padding boundaries")
                seq_ends = [packed_total] if max_seqlen == packed_total else [max_seqlen, packed_total]
                q, k, v = (t.view(packed_total, h, d) for t in recv)
                result = torch_npu.npu_fusion_attention(
                    q,
                    k,
                    v,
                    h,
                    input_layout="TND",
                    actual_seq_qlen=seq_ends,
                    actual_seq_kvlen=seq_ends,
                    scale=d**-0.5,
                    keep_prob=1.0,
                    sparse_mode=0,
                )[0]
                send = result.reshape(world, tokens, h, d).contiguous()
                out = buffers[bucket][1]
                self.output_ready[bucket].record(main)
                with torch.npu.stream(comm):
                    comm.wait_event(self.output_ready[bucket])
                    output_work = dist.all_to_all_single(out, send, group=self.group, async_op=True)
                outputs.append((out, output_work))
                keepalive.extend((q, k, v, result, send))
                if bucket:
                    total = contribute(bucket - 1, total)
            total = contribute(plan.buckets - 1, total)
            for tensor in keepalive:
                tensor.record_stream(main)
            result = project_once(full_output, m.out_proj.weight, m.out_proj.bias)
            self.done.record(main)
            self.used = True
            return result

    def close(self, *, restore_weights=True):
        """Run after device synchronization and DLO's CPU weight restoration."""
        if self._closed:
            return
        module = self.module
        if restore_weights:
            weight = self.layout.unpack_qkv(module.qkv_proj.weight.detach())
            bias = None if module.qkv_proj.bias is None else self.layout.unpack_qkv(module.qkv_proj.bias.detach())
            module.qkv_proj.weight.data = weight
            if bias is not None:
                module.qkv_proj.bias.data = bias
        if self._had_instance_forward:
            module.forward = self._original_forward
        else:
            del module.forward
        del module._h3_bucket_adapter
        self.workspace.close_after_synchronize()
        self._closed = True
