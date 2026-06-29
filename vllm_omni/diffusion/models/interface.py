# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, runtime_checkable

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
    """State-driven stage contract for diffusion step execution.

    The old ``prepare_encode`` / ``denoise_step`` / ``step_scheduler`` /
    ``post_decode`` names are intentionally not part of this protocol.  A
    step-capable pipeline exposes the composed stage names used by the runner:
    ``encode_stage`` / ``denoise_stage`` / ``scheduler_stage`` / ``decode_stage``.
    """

    supports_step_execution: ClassVar[bool] = True

    def encode_stage(self, state: "DiffusionRequestState") -> "DiffusionRequestState":
        """Run request-local validation, encoding, timestep and latent setup."""

    def denoise_stage(self, batch: "InputBatch") -> "torch.Tensor | None":
        """Run one batched DiT denoise forward."""

    def scheduler_stage(self, state: "DiffusionRequestState", noise: "torch.Tensor | None") -> None:
        """Advance one request-local scheduler step."""

    def decode_stage(self, state: "DiffusionRequestState") -> "DiffusionOutput":
        """Decode final latents into the public diffusion output."""


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


def supports_step_execution(pipeline: object) -> bool:
    """Return whether *pipeline* exposes the composed stage contract.

    This helper deliberately checks the new stage names and the explicit
    capability flag instead of accepting the old step-only method names.
    """

    if not bool(getattr(pipeline, "supports_step_execution", False)):
        return False
    return all(
        callable(getattr(pipeline, name, None))
        for name in ("encode_stage", "denoise_stage", "scheduler_stage", "decode_stage")
    )
