# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public chunked-transport surface used by the distributed layerwise hook."""
from __future__ import annotations

from ._chunk_pack import PinBudget, pack_local_shard, reconstruct_full_flat
from ._chunk_partition import build_part_manifest, is_chunk_transport_supported
from ._chunk_slots import ChunkedWeightTransport, TransferTicket
from ._chunk_types import (
    ChunkMeta,
    PartManifest,
    PinFailurePolicy,
    SlotPhase,
    TensorSpec,
    TransportBackendKind,
    WeightLayout,
    dtype_element_size,
)

__all__ = [
    "ChunkMeta",
    "ChunkedWeightTransport",
    "PartManifest",
    "PinBudget",
    "PinFailurePolicy",
    "SlotPhase",
    "TensorSpec",
    "TransferTicket",
    "TransportBackendKind",
    "WeightLayout",
    "build_part_manifest",
    "dtype_element_size",
    "is_chunk_transport_supported",
    "pack_local_shard",
    "reconstruct_full_flat",
]
