# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diffusion stage-kind and deployment-role helpers."""

from __future__ import annotations

from enum import Enum
from typing import Any


class DiffusionStageKind(str, Enum):
    INPUT_VALIDATION = "input_validation"
    TEXT_ENCODING = "text_encoding"
    CONDITION_ENCODING = "condition_encoding"
    LATENT_PREPARATION = "latent_preparation"
    TIMESTEP_PREPARATION = "timestep_preparation"
    DENOISING = "denoising"
    DECODING = "decoding"
    POSTPROCESS = "postprocess"


class DiffusionStageRole(str, Enum):
    MONOLITHIC = "monolithic"
    ENCODER = "encoder"
    DENOISER = "denoiser"
    DECODER = "decoder"


_ROLE_STAGE_KINDS: dict[DiffusionStageRole, tuple[DiffusionStageKind, ...]] = {
    DiffusionStageRole.MONOLITHIC: tuple(DiffusionStageKind),
    # Encoder owns the one-time request setup atoms; DiT and decode roles
    # consume the transport payloads produced after these atoms complete.
    DiffusionStageRole.ENCODER: (
        DiffusionStageKind.INPUT_VALIDATION,
        DiffusionStageKind.TEXT_ENCODING,
        DiffusionStageKind.CONDITION_ENCODING,
        DiffusionStageKind.LATENT_PREPARATION,
        DiffusionStageKind.TIMESTEP_PREPARATION,
    ),
    DiffusionStageRole.DENOISER: (DiffusionStageKind.DENOISING,),
    DiffusionStageRole.DECODER: (
        DiffusionStageKind.DECODING,
        DiffusionStageKind.POSTPROCESS,
    ),
}

_MODEL_STAGE_TO_ROLE: dict[str, DiffusionStageRole] = {
    # Accept both deployment-role names and older model_stage aliases so YAML
    # configs can migrate without changing every caller at once.
    "diffusion": DiffusionStageRole.MONOLITHIC,
    "monolithic": DiffusionStageRole.MONOLITHIC,
    "all": DiffusionStageRole.MONOLITHIC,
    "encode": DiffusionStageRole.ENCODER,
    "encoder": DiffusionStageRole.ENCODER,
    "text_encode": DiffusionStageRole.ENCODER,
    "condition_encode": DiffusionStageRole.ENCODER,
    "dit": DiffusionStageRole.DENOISER,
    "denoise": DiffusionStageRole.DENOISER,
    "denoiser": DiffusionStageRole.DENOISER,
    "decode": DiffusionStageRole.DECODER,
    "decoder": DiffusionStageRole.DECODER,
}


def normalize_diffusion_stage_role(value: Any) -> DiffusionStageRole:
    if isinstance(value, DiffusionStageRole):
        return value
    if value is None:
        return DiffusionStageRole.MONOLITHIC
    key = str(value).strip().lower()
    if key in _MODEL_STAGE_TO_ROLE:
        return _MODEL_STAGE_TO_ROLE[key]
    try:
        return DiffusionStageRole(key)
    except ValueError as exc:
        allowed = sorted({role.value for role in DiffusionStageRole} | set(_MODEL_STAGE_TO_ROLE))
        raise ValueError(f"Unknown diffusion stage role {value!r}; expected one of {allowed}") from exc


def diffusion_role_from_model_stage(model_stage: Any) -> DiffusionStageRole:
    return normalize_diffusion_stage_role(model_stage)


def stage_kinds_for_role(role: DiffusionStageRole | str | None) -> tuple[DiffusionStageKind, ...]:
    return _ROLE_STAGE_KINDS[normalize_diffusion_stage_role(role)]


def role_affinity(role: DiffusionStageRole | str | None, kind: DiffusionStageKind | str) -> bool:
    role = normalize_diffusion_stage_role(role)
    kind = kind if isinstance(kind, DiffusionStageKind) else DiffusionStageKind(str(kind))
    return kind in _ROLE_STAGE_KINDS[role]


def parse_diffusion_stage_kinds(value: Any, *, role: DiffusionStageRole | str | None = None) -> tuple[str, ...]:
    if value is None:
        return tuple(kind.value for kind in stage_kinds_for_role(role))
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.replace("|", ",").split(",")]
    else:
        raw_values = list(value)
    kinds = tuple(DiffusionStageKind(str(item).strip()).value for item in raw_values if str(item).strip())
    if not kinds:
        return tuple(kind.value for kind in stage_kinds_for_role(role))
    return kinds
