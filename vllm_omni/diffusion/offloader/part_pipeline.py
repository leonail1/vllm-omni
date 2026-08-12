# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage-3 compute-aware attention/MoE part pipeline (design sections 19-28).

Stage 1/2 stream one *whole block* per prefetch into two alternating device
slots.  Stage 3 splits every streamed block into two compute-phase parts —
``attention`` (used from block start to the mid-block boundary) and ``moe``
(used from the mid-block boundary to block end) — and gives each part its own
fixed device slot.  This adds two overlap layers on top of the Stage-1
chunk-level H2D/AllGather overlap:

* ``attention_i`` compute overlaps the ``moe_i`` prefetch (same block);
* ``moe_i`` compute overlaps the ``attention_(i+1)`` prefetch (next block).

Full-output staging therefore drops from ``2 * Bmax`` to about
``Amax + Emax`` (design section 22).

The model exposes the compute boundary through a split forward
(``forward_attention`` / ``forward_moe``) and declares the weight partition
with ``_block_weight_use_plan`` on the block class.  The transport engine
below never inspects attention/MoE module details; it only follows the
declared plan and the boundary calls.

Dynamic block skip is disabled while the part pipeline is active (design
section 24, policy 1): a skipped module never triggers the hooks that drive
its part prefetch/retire, so the collective sequence would diverge.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import os
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.platforms import current_omni_platform

from .cancellation import check_cancellation
from .chunked_transport import (
    BOUNDARY_BLOCK_END,
    BOUNDARY_BLOCK_START,
    BOUNDARY_MID_BLOCK,
    PART_ID_ATTENTION,
    PART_ID_MOE,
    BlockWeightUsePlan,
    ChunkedWeightTransport,
    PartManifest,
    SourceLayout,
    TensorSpec,
    TransferTicket,
    dtype_element_size,
)
from .tensor_utils import (
    is_materialized_tensor,
    make_offload_placeholder,
    set_tensor_storage,
)
from .weight_transport_backend import (
    ChunkCompletion,
    ChunkEvents,
    TransportStreams,
    WeightTransportBackend,
)

logger = init_logger(__name__)

# Same trace switch as the Stage-1/2 hook: markers are emitted only when
# VLLM_OMNI_DLO_TRACE is set.  Part markers extend the Stage-1 shape with an
# explicit part segment: dlo.<kind>.block_<id>.part_<pid>[.chunk_<id>].
_TRACE_ENV = "VLLM_OMNI_DLO_TRACE"

# Optional runtime boundary validation (design section 30.5): when set, the
# hook asserts part materialization state at every forward boundary.
_PART_VALIDATE_ENV = "VLLM_OMNI_DLO_PART_VALIDATE"


def _part_trace_marker(kind: str, block_id: int, part_id: str, chunk_id: int | None = None) -> str:
    base = f"dlo.{kind}.block_{block_id}.part_{part_id}"
    if chunk_id is None:
        return base
    return f"{base}.chunk_{chunk_id}"


# Shared no-op trace objects: with the trace switch off the hot path must
# not allocate a fresh nullcontext or closure per chunk.
_DISABLED_TRACE_RANGE = contextlib.nullcontext()


def _null_trace_factory(kind: str) -> Any:
    return _DISABLED_TRACE_RANGE


def _prefetch_barrier() -> None:
    """Sentinel queued on the single submit executor to drain prior work."""


# ---------------------------------------------------------------------- #
#  Plan resolution and tensor partitioning                                #
# ---------------------------------------------------------------------- #


def resolve_block_weight_use_plan(block_module: nn.Module) -> BlockWeightUsePlan | None:
    """Return the model-declared weight use plan for *block_module*, if any."""
    plan = getattr(type(block_module), "_block_weight_use_plan", None)
    if plan is None:
        return None
    if not isinstance(plan, BlockWeightUsePlan):
        raise TypeError(
            f"_block_weight_use_plan on {type(block_module).__name__} must be a "
            f"BlockWeightUsePlan, got {type(plan).__name__}"
        )
    return plan


def validate_block_weight_use_plan(plan: BlockWeightUsePlan) -> None:
    """Enforce the first-version plan shape (design sections 20-21).

    Version 1 supports exactly two non-overlapping parts: ``attention`` from
    block start to the mid-block boundary and ``moe`` from the mid-block
    boundary to block end.
    """
    if len(plan.parts) != 2:
        raise ValueError(
            "Stage-3 part pipeline v1 requires exactly two parts per block "
            f"(attention, moe), got {[spec.part_id for spec in plan.parts]}"
        )
    attention = plan.part(PART_ID_ATTENTION)
    moe = plan.part(PART_ID_MOE)
    if (attention.first_use_boundary, attention.last_use_boundary) != (
        BOUNDARY_BLOCK_START,
        BOUNDARY_MID_BLOCK,
    ):
        raise ValueError(
            "attention part must span block_start -> mid_block, got "
            f"{attention.first_use_boundary} -> {attention.last_use_boundary}"
        )
    if (moe.first_use_boundary, moe.last_use_boundary) != (
        BOUNDARY_MID_BLOCK,
        BOUNDARY_BLOCK_END,
    ):
        raise ValueError(
            f"moe part must span mid_block -> block_end, got {moe.first_use_boundary} -> {moe.last_use_boundary}"
        )
    overlap = set(attention.module_paths) & set(moe.module_paths)
    if overlap:
        raise ValueError(
            f"module paths declared in both attention and moe parts: {sorted(overlap)}; "
            "a parameter used in both phases must be declared resident, not packed twice"
        )


def split_specs_by_part(
    specs: Sequence[TensorSpec],
    plan: BlockWeightUsePlan,
) -> tuple[dict[str, list[TensorSpec]], list[TensorSpec]]:
    """Partition block tensor specs into per-part lists plus resident leftovers.

    Every spec whose name falls under a part's ``module_paths`` joins that
    part; everything else is returned as resident (kept on device, outside the
    chunk transport — the sanctioned handling for parameters read in both
    compute phases, design section 20).

    Raises when a declared module path matches no spec at all (a stale plan
    would silently stream a wrong partition), or when a part carries a dtype
    outside its ``allowed_dtypes``.
    """
    validate_block_weight_use_plan(plan)

    part_specs: dict[str, list[TensorSpec]] = {spec.part_id: [] for spec in plan.parts}
    resident_specs: list[TensorSpec] = []
    matched_paths: dict[str, int] = {path: 0 for part in plan.parts for path in part.module_paths}

    for name, tensor, is_buffer in specs:
        owner_part: str | None = None
        for part in plan.parts:
            for path in part.module_paths:
                if name == path or name.startswith(path + "."):
                    owner_part = part.part_id
                    matched_paths[path] += 1
                    break
            if owner_part is not None:
                break
        if owner_part is None:
            resident_specs.append((name, tensor, is_buffer))
        else:
            part_specs[owner_part].append((name, tensor, is_buffer))

    unmatched = [path for path, count in matched_paths.items() if count == 0]
    if unmatched:
        raise ValueError(f"weight use plan declares module paths with no tensors on this block: {unmatched}")

    for part in plan.parts:
        if not part.allowed_dtypes:
            continue
        bad = {tensor.dtype for name, tensor, _ in part_specs[part.part_id] if tensor.dtype not in part.allowed_dtypes}
        if bad:
            raise ValueError(
                f"part {part.part_id!r} carries dtypes outside its allowed_dtypes: "
                f"{[str(dtype) for dtype in sorted(bad, key=str)]}"
            )

    return part_specs, resident_specs


# ---------------------------------------------------------------------- #
#  Part-pipeline hook                                                     #
# ---------------------------------------------------------------------- #


class DistributedPartPipelineOffloadHook(ModelHook):
    """Per-block hook driving the Stage-3 attention/MoE part pipeline.

    Mirrors the Stage-1/2 ring: the hook attached to block ``i`` *produces*
    the parts of block ``i+1`` and drives the *consumption* of block ``i``'s
    parts (whose producer is the previous hook).  The two part slots are
    fixed: slot 0 always holds an attention part, slot 1 always an MoE part.

    Timeline for block ``i`` (design section 21):

    * ``pre_forward``: wait ``attention_i`` ready, submit ``moe_i`` prefetch;
    * ``new_forward``: compute ``attention_i``; at the mid-block boundary wait
      ``moe_i`` ready, retire ``attention_i``, submit ``attention_(i+1)``;
      compute ``moe_i``;
    * ``post_forward``: retire ``moe_i``.
    """

    _HOOK_NAME = "distributed_part_pipeline_offload"

    def __init__(
        self,
        next_block: nn.Module,
        device: torch.device,
        weight_shard_group: torch.distributed.ProcessGroup | None,
        weight_shard_size: int,
        weight_shard_rank: int,
        copy_stream: Any | None = None,
        comm_stream: Any | None = None,
        shared_buffers: list[dict[torch.dtype, torch.Tensor] | None] | None = None,
        shared_chunk_slot_events: list[Any | None] | None = None,
        shared_output_slot_events: list[Any | None] | None = None,
        shared_slot_owners: list[DistributedPartPipelineOffloadHook | None] | None = None,
        prepared_host_parts: dict[str, dict[str, Any]] | None = None,
        h2d_done_events: list[Any] | None = None,
        transport_done_events: list[Any] | None = None,
        relay_done_events: list[Any] | None = None,
        output_ready_events: list[Any] | None = None,
        last_use_events: list[Any | None] | None = None,
        prefetch_executor: concurrent.futures.Executor | None = None,
        data_transport_backend: WeightTransportBackend | None = None,
    ):
        assert isinstance(next_block, nn.Module), "transformer block must be type `torch.nn.Module`"
        if not prepared_host_parts:
            raise RuntimeError("part-pipeline hook requires backend-prepared Host parts")
        if set(prepared_host_parts) != {PART_ID_ATTENTION, PART_ID_MOE}:
            raise RuntimeError(
                f"part-pipeline hook requires attention+moe prepared parts, got {sorted(prepared_host_parts)}"
            )

        self.next_block = next_block
        self.device = device
        self.weight_shard_group = weight_shard_group
        self.weight_shard_size = weight_shard_size
        self.weight_shard_rank = weight_shard_rank

        if data_transport_backend is None:
            raise RuntimeError("data_transport_backend is required")
        self.data_transport_backend = data_transport_backend

        self.copy_stream = copy_stream or current_omni_platform.Stream()
        self.comm_stream = comm_stream or current_omni_platform.Stream()

        # Fixed part slots: attention -> 0, moe -> 1 (design section 22).
        self.part_slots: dict[str, int] = {PART_ID_ATTENTION: 0, PART_ID_MOE: 1}

        # Backend-prepared Host storage per part for the block this hook
        # transports (next_block).
        self.cpu_shards: dict[str, dict[torch.dtype, torch.Tensor]] = {}
        self.metadata: dict[str, dict[torch.dtype, list[dict[str, Any]]]] = {}
        self.manifests: dict[str, PartManifest] = {}
        self.fallback_reasons: dict[str, str | None] = {}
        self.part_transports: dict[str, ChunkedWeightTransport] = {}
        for part_id, prepared in prepared_host_parts.items():
            self.cpu_shards[part_id] = prepared["cpu_shards"]
            self.metadata[part_id] = prepared["metadata"]
            manifest: PartManifest = prepared["manifest"]
            self.manifests[part_id] = manifest
            self.fallback_reasons[part_id] = prepared.get("fallback_reason")
            transport = ChunkedWeightTransport(manifest.block_id, slot_count=2)
            transport.prepare(manifest, self.cpu_shards[part_id])
            self.part_transports[part_id] = transport

        self.block_id: int = self.manifests[PART_ID_ATTENTION].block_id
        # Block id used for the 'compute' marker, i.e. the module this hook is
        # attached to rather than the block it transports.  The backend
        # overwrites it right after construction.
        self.compute_block_id = self.block_id

        # Fixed part slots: either shared (from backend) or self-allocated.
        if shared_buffers is not None:
            self.gpu_buffers: list[dict[torch.dtype, torch.Tensor] | None] = shared_buffers
            self._owns_buffers = False
        else:
            self.gpu_buffers = [None, None]
            self._owns_buffers = True
        # Local chunk (AllGather input) buffers, indexed by *input* slot.
        self.gpu_shard_buffers: list[dict[torch.dtype, torch.Tensor] | None] = [None, None]

        self.ready_events: dict[str, Any | None] = {PART_ID_ATTENTION: None, PART_ID_MOE: None}
        self.ready_tickets: dict[str, TransferTicket | None] = {PART_ID_ATTENTION: None, PART_ID_MOE: None}

        self._output_ready_events: list[Any] = output_ready_events or [current_omni_platform.Event() for _ in range(2)]
        self._h2d_done_events: list[Any] = h2d_done_events or [current_omni_platform.Event() for _ in range(2)]
        self._transport_done_events: list[Any] = transport_done_events or [
            current_omni_platform.Event() for _ in range(2)
        ]
        self._relay_done_events: list[Any] = relay_done_events or []
        self._submit_dependency_event = current_omni_platform.Event()
        self._output_slot_events: list[Any | None] = (
            shared_output_slot_events if shared_output_slot_events is not None else [None, None]
        )
        self._chunk_slot_events: list[Any | None] = (
            shared_chunk_slot_events if shared_chunk_slot_events is not None else [None, None]
        )
        self._last_use_events: list[Any | None] = last_use_events if last_use_events is not None else [None, None]
        self._shared_slot_owners: list[DistributedPartPipelineOffloadHook | None] = (
            shared_slot_owners if shared_slot_owners is not None else [None, None]
        )

        self._request_generation = 0

        # Async submit bookkeeping per part.  Each hook has at most one
        # outstanding submission: its attention submit (mid-block boundary) is
        # drained by the consumer's pre_forward before its moe submit is
        # enqueued there.
        self._prefetch_executor = prefetch_executor
        self._part_futures: dict[str, concurrent.futures.Future | None] = {
            PART_ID_ATTENTION: None,
            PART_ID_MOE: None,
        }
        self._prefetched_parts: set[str] = set()

        self.trace_enabled = os.environ.get(_TRACE_ENV, "0") not in ("", "0", "false", "False")
        self._compute_trace: Any | None = None
        self._validate_boundaries = os.environ.get(_PART_VALIDATE_ENV, "0") not in ("", "0", "false", "False")

        # Backward link to previous hook (producer of this block's parts).
        self._prev_hook: DistributedPartPipelineOffloadHook | None = None

        # Group-first / group-tail markers for multi-DiT-group slot sharing.
        self._is_group_first: bool = False
        self._is_group_tail: bool = False
        self._group_id: int = -1
        self._shared_slot_group: list[int] | None = None  # [-1, -1], shared

        # Parameters/buffers of the current and next blocks.
        self.block_parameters: dict[str, nn.Parameter] = {}
        self.block_buffers: dict[str, torch.Tensor] = {}
        self.next_block_parameters: dict[str, nn.Parameter] = {}
        self.next_block_buffers: dict[str, torch.Tensor] = {}
        self._transported_names: dict[str, set[str]] = {
            part_id: {meta["name"] for metas in self.metadata[part_id].values() for meta in metas}
            for part_id in (PART_ID_ATTENTION, PART_ID_MOE)
        }

        self._cached_repoint: dict[str, list] | None = None
        self._repoint_views: dict[str, list] = {}

    # ------------------------------------------------------------------ #
    #  Initialization                                                      #
    # ------------------------------------------------------------------ #

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        module = super().initialize_hook(module)

        self.block_parameters = dict(module.named_parameters())
        self.block_buffers = dict(module.named_buffers())
        self.next_block_parameters = dict(self.next_block.named_parameters())
        self.next_block_buffers = dict(self.next_block.named_buffers())

        if self._owns_buffers:
            self._allocate_device_buffers()

        # Cache per-part parameter re-pointing metadata.
        self._cached_repoint = {}
        for part_id in (PART_ID_ATTENTION, PART_ID_MOE):
            repoint = []
            for dtype, metas in self.metadata[part_id].items():
                for m in metas:
                    target = (
                        self.next_block_parameters[m["name"]]
                        if m["name"] in self.next_block_parameters
                        else self.next_block_buffers[m["name"]]
                    )
                    repoint.append((target, dtype, m["offset"], m["numel"], m["shape"]))
            self._cached_repoint[part_id] = repoint
        # Lazily-built slot views (one per part, on first apply): the slot
        # buffers are persistent, so views are built once and reused, keeping
        # the hot path free of Tensor creation (design section 26.3).
        self._repoint_views: dict[str, list] = {}

        return module

    def _allocate_device_buffers(self) -> None:
        """Pre-allocate one device buffer per part slot."""
        for part_id, slot in self.part_slots.items():
            gpu_weights: dict[torch.dtype, torch.Tensor] = {}
            for dtype_manifest in self.manifests[part_id].dtypes:
                gpu_weights[dtype_manifest.dtype] = torch.empty(
                    dtype_manifest.padded_numel,
                    dtype=dtype_manifest.dtype,
                    device=self.device,
                )
            self.gpu_buffers[slot] = gpu_weights

    def is_part_materialized(self, part_id: str) -> bool:
        """Whether every transported tensor of *part_id* on this block is live."""
        producer = self._prev_hook
        names = producer._transported_names[part_id] if producer is not None else set(self.block_parameters)
        for name in names:
            target = self.block_parameters.get(name, self.block_buffers.get(name))
            if target is not None and not is_materialized_tensor(target):
                return False
        return True

    # ------------------------------------------------------------------ #
    #  Tracing and boundary validation                                     #
    # ------------------------------------------------------------------ #

    def _trace_range(self, kind: str, part_id: str, chunk_id: int | None = None, block_id: int | None = None) -> Any:
        if not self.trace_enabled:
            return _DISABLED_TRACE_RANGE
        marker_block = self.block_id if block_id is None else block_id
        return torch.profiler.record_function(_part_trace_marker(kind, marker_block, part_id, chunk_id))

    def validate_block_part_state(self, part_id: str, *, materialized: bool) -> None:
        """Assert every transported tensor of *part_id* on this block is
        materialized (or placeholder) — the model-level first-use/last-use
        instrumentation of design section 30.5."""
        producer = self._prev_hook
        names = producer._transported_names[part_id] if producer is not None else set(self.block_parameters)
        for name in names:
            target = self.block_parameters.get(name, self.block_buffers.get(name))
            if target is None:
                continue
            live = is_materialized_tensor(target)
            if live != materialized:
                state = "materialized" if live else "placeholder"
                raise RuntimeError(
                    f"part boundary violation on block {self.compute_block_id} part {part_id!r}: "
                    f"tensor {name!r} is {state}, expected "
                    f"{'materialized' if materialized else 'placeholder'}"
                )

    # ------------------------------------------------------------------ #
    #  Prefetch: chunked H2D + AllGather for one part                     #
    # ------------------------------------------------------------------ #

    @torch.compiler.disable
    def _submit_part_prefetch(
        self,
        part_id: str,
        non_blocking: bool = True,
        submit_after_event: Any | None = None,
    ) -> bool:
        """Enqueue the chunked H2D + AllGather pipeline for one part.

        Same worker-thread contract as the Stage-1/2 submit: stream work only,
        no Python-level Parameter mutation.  Returns True when a new
        submission was made.
        """
        if self.fallback_reasons[part_id] is not None:
            # Pageable Host memory: an async copy would race with the packer.
            non_blocking = False

        if submit_after_event is None:
            raise RuntimeError("part prefetch submit requires a main-thread dependency event")
        self.copy_stream.wait_event(submit_after_event)

        slot = self.part_slots[part_id]
        manifest = self.manifests[part_id]
        transport = self.part_transports[part_id]
        gpu_weights = self.gpu_buffers[slot]
        assert gpu_weights is not None, f"gpu_buffers[{slot}] not allocated"

        previous_ticket = self.ready_tickets[part_id]
        if previous_ticket is not None and transport.state.is_current(previous_ticket):
            # Idempotent re-prefetch only within the same generation.  A live
            # ticket from another generation owns the slot with a different
            # key and must fail fast (design section 25), not be returned.
            if previous_ticket.request_generation != self._request_generation:
                raise RuntimeError(
                    f"output slot {slot} holds a stale generation: ticket generation "
                    f"{previous_ticket.request_generation} vs current {self._request_generation}"
                )
            return False

        previous_owner = self._shared_slot_owners[slot]
        if previous_owner is not None and previous_owner is not self:
            overwritten_ticket = previous_owner.ready_tickets[part_id]
            if overwritten_ticket is not None and previous_owner.part_transports[part_id].state.is_current(
                overwritten_ticket
            ):
                previous_owner.part_transports[part_id].record_last_use(
                    overwritten_ticket,
                    self._output_slot_events[slot],
                )
                previous_owner.ready_tickets[part_id] = None
                previous_owner.ready_events[part_id] = None

        evt = self._output_ready_events[slot]
        ticket = transport.begin_submission(
            output_slot=slot,
            request_generation=self._request_generation,
            ready_event=evt,
            part_id=part_id,
            backend_id=self.data_transport_backend.kind.value,
            # Design section 24: the canonical collective key carries the
            # forward generation, block, part, slot and manifest digest.
            last_collective_key=(
                self._request_generation,
                self.block_id,
                part_id,
                slot,
                manifest.digest[:16],
            ),
        )
        self.ready_tickets[part_id] = ticket

        cpu_shards = self.cpu_shards[part_id]
        if non_blocking:
            for dtype, cpu_source in cpu_shards.items():
                if cpu_source.numel() and not cpu_source.is_pinned():
                    # Pre-collective validation failure: the slot is still in
                    # PREFETCH_ENQUEUED, so cancel instead of poisoning the
                    # process group (design sections 25/27.1).
                    transport.cancel_submission(
                        ticket,
                        f"chunk H2D requires pinned Host source for block {self.block_id} "
                        f"part {part_id!r} dtype={dtype}; tensor is not pinned.",
                    )
                    self.ready_tickets[part_id] = None
                    raise RuntimeError(
                        f"chunk H2D requires pinned Host source for block {self.block_id} "
                        f"part {part_id!r} dtype={dtype}; tensor is not pinned."
                    )

        streams = TransportStreams(copy=self.copy_stream, communication=self.comm_stream)
        shard_bufs = self.gpu_shard_buffers

        chunk_specs = [
            (dtype_manifest.dtype, chunk) for dtype_manifest in manifest.dtypes for chunk in dtype_manifest.chunks
        ]
        try:
            self.data_transport_backend.begin_part(streams, self._output_slot_events[slot])
            completions: list[ChunkCompletion] = []
            trace_factory = None if self.trace_enabled else _null_trace_factory
            for transfer_index, (dtype, chunk) in enumerate(chunk_specs):
                input_slot = transfer_index % 2
                cpu_source = cpu_shards[dtype]
                if manifest.source_layout is SourceLayout.FS_SHARDED_HOST:
                    source = cpu_source[chunk.cpu_offset : chunk.cpu_offset + chunk.local_numel]
                elif cpu_source.numel():
                    source = cpu_source[chunk.full_offset : chunk.full_offset + chunk.padded_numel]
                else:
                    source = None

                local_input = None
                if self.data_transport_backend.requires_local_input:
                    if len(shard_bufs) != 2 or shard_bufs[0] is None or shard_bufs[1] is None:
                        raise RuntimeError("selected transport backend requires two local chunk input buffers")
                    local_input = shard_bufs[input_slot][dtype][: chunk.local_numel]

                completion = self.data_transport_backend.submit_chunk(
                    source=source,
                    local_input=local_input,
                    full_output=gpu_weights[dtype][chunk.full_offset : chunk.full_offset + chunk.padded_numel],
                    chunk_meta=chunk,
                    streams=streams,
                    events=ChunkEvents(
                        h2d_done=self._h2d_done_events[input_slot],
                        transport_done=self._transport_done_events[input_slot],
                        input_reusable=self._chunk_slot_events[input_slot],
                        relay_done=self._relay_done_events[input_slot] if self._relay_done_events else None,
                    ),
                    group=self.weight_shard_group,
                    generation=self._request_generation,
                    non_blocking=non_blocking,
                    trace=(
                        trace_factory
                        if trace_factory is not None
                        else lambda kind, chunk_id=chunk.chunk_id: self._trace_range(kind, part_id, chunk_id)
                    ),
                )
                completions.append(completion)
                if self.data_transport_backend.requires_local_input:
                    self._chunk_slot_events[input_slot] = completion.event

            self.data_transport_backend.finalize_part(
                completions,
                ready_event=evt,
                streams=streams,
            )
        except BaseException as exc:
            # A failure inside the submission region may leave some ranks
            # mid-collective: the process group can no longer be trusted
            # (design sections 25/27.2).  Record the generation and the last
            # collective key for post-mortem diagnosis (section 27.2 item 1).
            transport.poison(
                f"block {self.block_id} part {part_id!r} submission failed "
                f"(request_generation={self._request_generation}, "
                f"last_collective_key={ticket.last_collective_key}): {exc!r}"
            )
            raise

        self.ready_events[part_id] = evt
        transport.mark_ready(ticket)
        self._prefetched_parts.add(part_id)
        self._shared_slot_owners[slot] = self

        if self.trace_enabled:
            # Per-prefetch structured record (design section 28).
            full_bytes = sum(
                dtype_manifest.padded_numel * dtype_element_size(dtype_manifest.dtype)
                for dtype_manifest in manifest.dtypes
            )
            logger.info(
                "DLO_PREFETCH gen=%d shard=%d/%d block=%d part=%s backend=%s layout=%s "
                "full_bytes=%d local_h2d_bytes=%d chunks=%d slot=%d fallback=%s last_key=%s",
                self._request_generation,
                self.weight_shard_rank,
                self.weight_shard_size,
                self.block_id,
                part_id,
                self.data_transport_backend.kind.value,
                manifest.source_layout.value,
                full_bytes,
                manifest.pinned_bytes,
                manifest.chunk_count,
                slot,
                self.fallback_reasons[part_id],
                ticket.last_collective_key,
            )

        if self._shared_slot_group is not None:
            self._shared_slot_group[slot] = self._group_id
        return True

    @torch.compiler.disable
    def _apply_part_repoint(self, part_id: str) -> None:
        """Point next block's part Parameters at the slot's device buffer."""
        if self._cached_repoint is None:
            return
        slot = self.part_slots[part_id]
        gpu_weights = self.gpu_buffers[slot]
        if gpu_weights is None:
            return
        views = self._repoint_views.get(part_id)
        if views is None:
            views = [
                (target, gpu_weights[dtype][offset : offset + numel].view(shape))
                for target, dtype, offset, numel, shape in self._cached_repoint[part_id]
            ]
            self._repoint_views[part_id] = views
        for target, view in views:
            set_tensor_storage(target, view)
        # READY -> BOUND: the consumer's Parameters now view the slot's
        # storage (design section 25).  No-op when already bound.
        ticket = self.ready_tickets[part_id]
        transport = self.part_transports[part_id]
        if ticket is not None and transport.state.is_current(ticket):
            transport.mark_bound(ticket)

    @torch.compiler.disable
    def prefetch_part_layer(self, part_id: str, non_blocking: bool = True) -> None:
        """Synchronous part prefetch for bootstrap and repair paths."""
        # Never let a main-thread submit interleave with a worker submit: every
        # FS rank must issue the identical collective sequence.
        self._drain_part(part_id, apply_repoint=True)
        compute_stream = current_omni_platform.current_stream()
        self._submit_dependency_event.record(compute_stream)
        with self._trace_range("submit", part_id):
            if self._prefetch_executor is not None:
                submitted = self._prefetch_executor.submit(
                    self._submit_part_prefetch,
                    part_id,
                    non_blocking,
                    self._submit_dependency_event,
                ).result()
            else:
                submitted = self._submit_part_prefetch(
                    part_id,
                    non_blocking,
                    submit_after_event=self._submit_dependency_event,
                )
        if submitted:
            self._apply_part_repoint(part_id)

    def _drain_part(self, part_id: str, apply_repoint: bool = True) -> None:
        """Wait for this hook's outstanding async submit of *part_id*."""
        future = self._part_futures[part_id]
        if future is None:
            return
        self._part_futures[part_id] = None
        submitted = future.result()
        if apply_repoint and part_id in self._prefetched_parts and submitted is not False:
            self._apply_part_repoint(part_id)

    def get_part_weights(self, part_id: str) -> dict[torch.dtype, torch.Tensor] | None:
        """Attach the producer's ready event for *part_id* to the compute stream.

        A missing or non-current ticket fails fast: waiting on the raw event
        anyway would reintroduce exactly the read-after-overwrite the ticket
        state machine exists to prevent.
        """
        producer = self._prev_hook if self._prev_hook is not None else self
        producer._drain_part(part_id, apply_repoint=True)

        evt = producer.ready_events[part_id]
        if evt is None:
            ticket = producer.ready_tickets[part_id]
            phase = producer.part_transports[part_id].state.current_phase(ticket) if ticket is not None else None
            logger.warning(
                "DLO part-pipeline consume failure debug: consumer_block=%s producer_block=%s part=%s "
                "producer_ticket=%s phase=%s prefetched=%s slot_owners=%s slot_group=%s group_id=%s "
                "request_generation=%s",
                self.block_id,
                producer.block_id,
                part_id,
                ticket,
                phase,
                producer._prefetched_parts,
                [type(o).__name__ if o is not None else None for o in self._shared_slot_owners],
                self._shared_slot_group,
                self._group_id,
                self._request_generation,
            )
            raise RuntimeError(
                f"block {self.block_id} has no ready event for part {part_id!r}: "
                "compute would read weights before any transfer published them"
            )

        ticket = producer.ready_tickets[part_id]
        transport = producer.part_transports[part_id]
        if ticket is None or not transport.state.is_current(ticket):
            raise RuntimeError(
                f"block {self.block_id} cannot consume part {part_id!r}: the producing "
                f"ticket is {'missing' if ticket is None else 'stale'} because another "
                "transfer took the slot; refusing to wait on an event that no longer "
                "proves this part's weights are ready"
            )
        # Trace the consumer end of the happens-before chain (design section
        # 28): attach == the part is ready and compute is about to read it.
        with self._trace_range("attach", part_id):
            transport.attach_ready(
                ticket,
                lambda event: current_omni_platform.current_stream().wait_event(event),
            )
        return self.gpu_buffers[self.part_slots[part_id]]

    # ------------------------------------------------------------------ #
    #  Last-use / retire                                                  #
    # ------------------------------------------------------------------ #

    @torch.compiler.disable
    def _retire_part(self, part_id: str) -> None:
        """Record this block's last use of *part_id* and release the slot.

        Runs on the main thread right after the consuming compute phase was
        enqueued; the recorded event captures that phase's kernels on the
        compute stream.
        """
        slot = self.part_slots[part_id]
        compute_stream = current_omni_platform.current_stream()

        evt = self._last_use_events[slot]
        if evt is None:
            evt = current_omni_platform.Event()
            self._last_use_events[slot] = evt
        evt.record(compute_stream)
        self._output_slot_events[slot] = evt

        prev = self._prev_hook
        if prev is not None:
            ticket = prev.ready_tickets[part_id]
            if ticket is not None and prev.part_transports[part_id].state.is_current(ticket):
                # Trace the last-use edge of the happens-before chain
                # (design section 28).
                with self._trace_range("retire", part_id):
                    prev.part_transports[part_id].record_last_use(ticket, evt)
                prev.ready_tickets[part_id] = None
                prev.ready_events[part_id] = None
                if self._shared_slot_owners[slot] is prev:
                    self._shared_slot_owners[slot] = None

        transported_names = prev._transported_names[part_id] if prev is not None else set(self.block_parameters)
        for name in transported_names:
            target = self.block_parameters.get(name, self.block_buffers.get(name))
            if target is not None:
                set_tensor_storage(target, make_offload_placeholder(target))

    # ------------------------------------------------------------------ #
    #  Request lifecycle                                                  #
    # ------------------------------------------------------------------ #

    @torch.compiler.disable
    def retire_unconsumed_prefetches(self) -> None:
        """Retire live tickets whose target block will not run this step.

        Design section 24, policy-2 reconciliation: a cache-dit skip decision
        is made after block 0's forward, but block 0's mid-block boundary
        already submitted the attention prefetch for block 1.  That
        prefetch's data cannot survive until the next compute step — the next
        step's repair prefetch for block 0's own attention part reuses the
        shared slot first — so the ticket is retired here as a never-consumed
        transfer (READY/BOUND retire, design section 25) and the target
        block's Parameters are reset to placeholders.  The next compute step
        re-prefetches via the existing repair path.
        """
        for part_id in (PART_ID_ATTENTION, PART_ID_MOE):
            self._drain_part(part_id, apply_repoint=False)
            ticket = self.ready_tickets[part_id]
            if ticket is None:
                continue
            transport = self.part_transports[part_id]
            if not transport.state.is_current(ticket):
                continue
            logger.debug(
                "DLO skip reconciliation: retiring unconsumed prefetch block=%s part=%s ticket=%s",
                self.block_id,
                part_id,
                ticket.validation_key,
            )
            transport.record_last_use(ticket, self.ready_events[part_id])
            self.ready_tickets[part_id] = None
            self.ready_events[part_id] = None
            slot = self.part_slots[part_id]
            if self._shared_slot_owners[slot] is self:
                self._shared_slot_owners[slot] = None
            for name in self._transported_names[part_id]:
                target = self.next_block_parameters.get(name, self.next_block_buffers.get(name))
                if target is not None:
                    set_tensor_storage(target, make_offload_placeholder(target))
            self._prefetched_parts.discard(part_id)

    def set_request_generation(self, generation: int) -> None:
        if generation < self._request_generation:
            raise RuntimeError(f"request generation moved backwards: {generation} < {self._request_generation}")
        self._request_generation = generation

    def drain_request(self) -> None:
        """Release every live part slot so the next request starts clean."""
        for part_id in (PART_ID_ATTENTION, PART_ID_MOE):
            self._drain_part(part_id, apply_repoint=False)

        if self._compute_trace is not None:
            self._compute_trace.__exit__(None, None, None)
            self._compute_trace = None

        for part_id in (PART_ID_ATTENTION, PART_ID_MOE):
            ticket = self.ready_tickets[part_id]
            transport = self.part_transports[part_id]
            if ticket is None or not transport.state.is_current(ticket):
                continue
            # A tail prefetch issued by the final block has no consumer in
            # this request.  Retire it at producer-ready without wiring that
            # unused transfer into the default compute stream.
            transport.record_last_use(ticket, self.ready_events[part_id])
            self.ready_tickets[part_id] = None
            self.ready_events[part_id] = None
            slot = self.part_slots[part_id]
            if self._shared_slot_owners[slot] is self:
                self._shared_slot_owners[slot] = None

        for part_id in (PART_ID_ATTENTION, PART_ID_MOE):
            for name in self._transported_names[part_id]:
                target = self.next_block_parameters.get(name, self.next_block_buffers.get(name))
                if target is not None:
                    set_tensor_storage(target, make_offload_placeholder(target))

        self._prefetched_parts.clear()
        for transport in self.part_transports.values():
            transport.reset()

    # ------------------------------------------------------------------ #
    #  ModelHook interface                                                #
    # ------------------------------------------------------------------ #

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        # Drain the producer's attention submit (issued at the previous
        # block's mid-block boundary) and bind this block's attention part.
        if self._prev_hook is not None:
            self._prev_hook._drain_part(PART_ID_ATTENTION, apply_repoint=True)

        compute_stream = current_omni_platform.current_stream()

        # Group-first hook: the attention slot may have been overwritten by
        # another DiT group since our last forward.  Re-prefetch synchronously
        # unless the slot still holds our own group's data.
        if self._is_group_first and self._prev_hook is not None:
            # Design section 27.3: abort a client-cancelled request at this
            # uniform forward boundary (FS-group vote inside).
            check_cancellation()
            if self._prefetch_executor is not None:
                self._prefetch_executor.submit(_prefetch_barrier).result()
            slot_contaminated = True
            attn_slot = self.part_slots[PART_ID_ATTENTION]
            if self._shared_slot_group is not None:
                slot_contaminated = self._shared_slot_group[attn_slot] != self._group_id
            ticket = self._prev_hook.ready_tickets[PART_ID_ATTENTION]
            has_current_ticket = ticket is not None and self._prev_hook.part_transports[
                PART_ID_ATTENTION
            ].state.is_current(ticket)
            if slot_contaminated:
                if has_current_ticket:
                    self._prev_hook.part_transports[PART_ID_ATTENTION].record_last_use(
                        ticket, self._output_slot_events[attn_slot]
                    )
                    self._prev_hook.ready_tickets[PART_ID_ATTENTION] = None
                self._prev_hook.prefetch_part_layer(PART_ID_ATTENTION, non_blocking=False)
            elif not has_current_ticket:
                # The slot still carries our group's marker, but a cache-skip
                # pass never ran the tail block, so no attention prefetch for
                # this block was ever submitted (design section 24, policy 2).
                # The non-contaminated fast path may not assume a ticket
                # exists: repair with a synchronous prefetch.
                self._prev_hook.prefetch_part_layer(PART_ID_ATTENTION, non_blocking=False)
        elif not self.is_part_materialized(PART_ID_ATTENTION) and self._prev_hook is not None:
            # No valid async attention prefetch is in flight for this block.
            ticket = self._prev_hook.ready_tickets[PART_ID_ATTENTION]
            has_current_ticket = ticket is not None and self._prev_hook.part_transports[
                PART_ID_ATTENTION
            ].state.is_current(ticket)
            if not has_current_ticket:
                self._prev_hook.prefetch_part_layer(PART_ID_ATTENTION, non_blocking=False)

        # Wait attention_i ready (attaches the producer's ready event).
        self.get_part_weights(PART_ID_ATTENTION)
        if self._validate_boundaries:
            self.validate_block_part_state(PART_ID_ATTENTION, materialized=True)
            self.validate_block_part_state(PART_ID_MOE, materialized=False)

        # Submit moe_i's prefetch off-thread so it overlaps attention compute.
        producer = self._prev_hook if self._prev_hook is not None else self
        producer._drain_part(PART_ID_MOE, apply_repoint=True)
        producer._submit_dependency_event.record(compute_stream)
        with self._trace_range("submit", PART_ID_MOE, block_id=producer.block_id):
            if self._prefetch_executor is not None:
                producer._part_futures[PART_ID_MOE] = self._prefetch_executor.submit(
                    producer._submit_part_prefetch,
                    PART_ID_MOE,
                    True,
                    producer._submit_dependency_event,
                )
            else:
                submitted = producer._submit_part_prefetch(
                    PART_ID_MOE,
                    True,
                    producer._submit_dependency_event,
                )
                if submitted:
                    producer._apply_part_repoint(PART_ID_MOE)

        if self._compute_trace is not None:
            raise RuntimeError("compute trace range was not closed")
        if self.trace_enabled:
            self._compute_trace = self._trace_range("compute", PART_ID_ATTENTION, block_id=self.compute_block_id)
            self._compute_trace.__enter__()

        return args, kwargs

    def new_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> Any:
        # --- attention phase (design section 23, step 3) ---
        attn_out = module.forward_attention(*args, **kwargs)

        if self._compute_trace is not None:
            self._compute_trace.__exit__(None, None, None)
            self._compute_trace = None

        # --- mid-block boundary (steps 4-7) ---
        compute_stream = current_omni_platform.current_stream()

        # Wait moe_i ready and bind its parameters.
        self.get_part_weights(PART_ID_MOE)
        if self._validate_boundaries:
            self.validate_block_part_state(PART_ID_MOE, materialized=True)

        # Retire attention_i: record last-use on the compute stream (captures
        # the attention kernels), release the ticket, detach the storage.
        self._retire_part(PART_ID_ATTENTION)
        if self._validate_boundaries:
            self.validate_block_part_state(PART_ID_ATTENTION, materialized=False)

        # Submit attention_(i+1) into the attention slot; it overlaps moe_i
        # compute.  This hook is the producer of the next block.
        self._drain_part(PART_ID_ATTENTION, apply_repoint=True)
        self._submit_dependency_event.record(compute_stream)
        with self._trace_range("submit", PART_ID_ATTENTION):
            if self._prefetch_executor is not None:
                self._part_futures[PART_ID_ATTENTION] = self._prefetch_executor.submit(
                    self._submit_part_prefetch,
                    PART_ID_ATTENTION,
                    True,
                    self._submit_dependency_event,
                )
            else:
                submitted = self._submit_part_prefetch(
                    PART_ID_ATTENTION,
                    True,
                    self._submit_dependency_event,
                )
                if submitted:
                    self._apply_part_repoint(PART_ID_ATTENTION)

        if self.trace_enabled:
            self._compute_trace = self._trace_range("compute", PART_ID_MOE, block_id=self.compute_block_id)
            self._compute_trace.__enter__()

        # --- moe phase (step 8) ---
        return module.forward_moe(attn_out, *args, **kwargs)

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        if self._compute_trace is not None:
            self._compute_trace.__exit__(None, None, None)
            self._compute_trace = None
        # Retire moe_i (design section 23, step 9).
        self._retire_part(PART_ID_MOE)
        return output


def apply_part_pipeline_block_hook(
    module: nn.Module,
    next_block: nn.Module,
    device: torch.device,
    weight_shard_group: torch.distributed.ProcessGroup | None,
    weight_shard_size: int,
    weight_shard_rank: int,
    copy_stream: Any | None = None,
    comm_stream: Any | None = None,
    shared_buffers: list[dict[torch.dtype, torch.Tensor] | None] | None = None,
    shared_chunk_slot_events: list[Any | None] | None = None,
    shared_output_slot_events: list[Any | None] | None = None,
    shared_slot_owners: list[DistributedPartPipelineOffloadHook | None] | None = None,
    prepared_host_parts: dict[str, dict[str, Any]] | None = None,
    h2d_done_events: list[Any] | None = None,
    transport_done_events: list[Any] | None = None,
    relay_done_events: list[Any] | None = None,
    output_ready_events: list[Any] | None = None,
    last_use_events: list[Any | None] | None = None,
    prefetch_executor: concurrent.futures.Executor | None = None,
    data_transport_backend: WeightTransportBackend | None = None,
) -> DistributedPartPipelineOffloadHook:
    """Register a DistributedPartPipelineOffloadHook on *module*."""
    registry = HookRegistry.get_or_create(module)
    hook = DistributedPartPipelineOffloadHook(
        next_block=next_block,
        device=device,
        weight_shard_group=weight_shard_group,
        weight_shard_size=weight_shard_size,
        weight_shard_rank=weight_shard_rank,
        copy_stream=copy_stream,
        comm_stream=comm_stream,
        shared_buffers=shared_buffers,
        shared_chunk_slot_events=shared_chunk_slot_events,
        shared_output_slot_events=shared_output_slot_events,
        shared_slot_owners=shared_slot_owners,
        prepared_host_parts=prepared_host_parts,
        h2d_done_events=h2d_done_events,
        transport_done_events=transport_done_events,
        relay_done_events=relay_done_events,
        output_ready_events=output_ready_events,
        last_use_events=last_use_events,
        prefetch_executor=prefetch_executor,
        data_transport_backend=data_transport_backend,
    )
    registry.register_hook(DistributedPartPipelineOffloadHook._HOOK_NAME, hook)
    return hook


def remove_part_pipeline_block_hook(module: nn.Module) -> None:
    """Remove the part-pipeline offload hook from *module*."""
    registry: HookRegistry | None = getattr(module, "_hook_registry", None)
    if registry is not None:
        registry.remove_hook(DistributedPartPipelineOffloadHook._HOOK_NAME)
        logger.debug("Removed part-pipeline offload hook from %s", module.__class__.__name__)
