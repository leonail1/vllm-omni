# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Protocol,
    runtime_checkable,
)

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
class DiffusionAtoms(Protocol):
    """Request lifecycle atoms implemented by a diffusion pipeline."""

    supports_step_execution: ClassVar[bool] = True

    def init_state(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Initialize pipeline-private request state."""
        ...

    def check_inputs(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Validate request inputs before model work begins."""
        ...

    def encode(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Encode request inputs."""
        ...

    def prepare(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Prepare latents, timesteps, and denoising inputs."""
        ...

    def diffuse(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Run the request-mode diffusion loop."""
        ...

    def decode(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Decode the final latents."""
        ...

    def postprocess(self, state: DiffusionRequestState) -> DiffusionOutput:
        """Build the public diffusion output."""
        ...


@runtime_checkable
class DiffusionStepHooks(Protocol):
    """Step-batching hooks implemented by a diffusion pipeline."""

    def build_step_batch(
        self,
        states: list[DiffusionRequestState],
        *,
        cached_batch: InputBatch | None = None,
    ) -> InputBatch:
        """Build the model batch for one scheduler tick."""
        ...

    def build_step_attention_metadata(
        self,
        input_batch: InputBatch,
    ) -> object | None:
        """Build optional attention metadata for the step batch."""
        ...

    def denoise_step(self, input_batch: InputBatch, **kwargs: Any) -> torch.Tensor | None:
        """Run one denoise forward on a runner-assembled batch."""
        ...

    def step_scheduler(
        self,
        state: DiffusionRequestState,
        noise_pred: torch.Tensor,
        **kwargs: Any,
    ) -> DiffusionRequestState:
        """Apply one scheduler step to request-local state."""
        ...


@runtime_checkable
class SupportsStepExecution(Protocol):
    """State-driven step-level execution protocol for diffusion pipelines.

    Pipelines should split request-level ``forward()`` into:
    ``prepare_encode()`` (one-time request setup), ``denoise_step()``
    (one denoise forward), ``step_scheduler()`` (one scheduler update),
    and ``post_decode()`` (final decode).
    """

    supports_step_execution: ClassVar[bool] = True

    def prepare_encode(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionRequestState:
        """Prepare request-level inputs and return initialized state."""
        ...

    def denoise_step(self, input_batch: InputBatch, **kwargs: Any) -> torch.Tensor | None:
        """Run one denoise forward on the runner-assembled batch."""
        ...

    def step_scheduler(
        self,
        state: DiffusionRequestState,
        noise_pred: torch.Tensor,
        **kwargs: Any,
    ) -> DiffusionRequestState:
        """Run one scheduler step."""
        ...

    def post_decode(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionOutput:
        """Decode output after denoise loop or at a partial chunk boundary."""
        ...


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
    """Return whether `pipeline` implements :class:`SupportsStepExecution`."""

    return isinstance(pipeline, SupportsStepExecution)
