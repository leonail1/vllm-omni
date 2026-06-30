# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, TypeGuard, runtime_checkable

if TYPE_CHECKING:
    import torch

    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import DiffusionRequestState


@runtime_checkable
class SupportImageInput(Protocol):
    support_image_input: ClassVar[bool] = True
    color_format: ClassVar[str] = "RGB"  # Default color format


@dataclass(frozen=True)
class ReferenceVideoDecodeSpec:
    max_frames: int | None = None
    keep: Literal["first", "last"] = "first"


@runtime_checkable
class SupportAudioInput(Protocol):
    support_audio_input: ClassVar[bool] = True


@runtime_checkable
class SupportAudioOutput(Protocol):
    support_audio_output: ClassVar[bool] = True


@runtime_checkable
class SupportsStepExecution(Protocol):
    """Stage-composed execution contract for diffusion pipelines."""

    supports_step_execution: ClassVar[bool] = True

    def encode_stage(self, state: "DiffusionRequestState") -> "DiffusionRequestState": ...

    def denoise_stage(self, batch: "InputBatch") -> "torch.Tensor | None": ...

    def scheduler_stage(self, state: "DiffusionRequestState", noise: "torch.Tensor | None") -> None: ...

    def decode_stage(self, state: "DiffusionRequestState") -> "DiffusionOutput": ...


@runtime_checkable
class SupportsComponentDiscovery(Protocol):
    """Declares which submodules serve as pipeline components.

    Used by the framework to locate DiT, encoder, and VAE modules for
    CPU offload, HSDP sharding, and other operations that need to know
    the pipeline's internal structure.

    All attribute names support dotted paths for nested submodules
    (e.g. ``"pipe.transformer"``).

    Attributes:
        _dit_modules: Denoising submodules (on GPU during diffusion).
        _encoder_modules: Encoder submodules (offloaded during diffusion).
        _vae_modules: VAE(s) (always on GPU).
        _resident_modules: Extra modules pinned on GPU during layerwise
            offloading.  Optional, defaults to ``[]``.
    """

    _dit_modules: ClassVar[list[str]]
    _encoder_modules: ClassVar[list[str]]
    _vae_modules: ClassVar[list[str]]
    _resident_modules: ClassVar[list[str]] = []


def supports_step_execution(pipeline: object) -> TypeGuard[SupportsStepExecution]:
    """Return whether *pipeline* exposes the composed stage contract.

    ``@runtime_checkable`` protocols only validate attribute presence, so this
    helper also checks the explicit capability flag and that stage entries are
    callable.
    """

    if not isinstance(pipeline, SupportsStepExecution):
        return False
    if not bool(getattr(pipeline, "supports_step_execution", False)):
        return False

    return all(
        callable(getattr(pipeline, name, None))
        for name in ("encode_stage", "denoise_stage", "scheduler_stage", "decode_stage")
    )
