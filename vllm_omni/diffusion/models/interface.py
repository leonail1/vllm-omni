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
    from vllm_omni.diffusion.request import OmniDiffusionRequest
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
class SupportsDiffusionAtoms(Protocol):
    """Atomized diffusion pipeline contract used by the step/batch runner.

    ``forward(req)`` remains the request-mode golden path. Step/batch execution
    uses these atoms so the runner can own state lifetime, batching,
    scatter/gather, and stage transport while the pipeline owns model math.
    """

    supports_diffusion_atoms: ClassVar[bool] = True
    supports_varlen_batch: ClassVar[bool] = False
    supports_stage_split: ClassVar[bool] = False

    def forward(self, req: OmniDiffusionRequest, **kwargs: Any) -> DiffusionOutput:
        """Run the request-mode golden path."""

    def init_state(self, req: OmniDiffusionRequest) -> DiffusionRequestState:
        """Create a request state for runner-owned step execution."""

    def validation(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Validate request inputs and normalize request-local arguments."""

    def encoding(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Encode text/image/audio conditioning into model-private state."""

    def preparation(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Prepare latents, timesteps, scheduler state, and runner metadata."""

    def denoising(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Run the full request-mode denoising loop over one state."""

    def decoding(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Decode final latents into a pipeline-private output payload."""

    def postprocess(self, state: DiffusionRequestState) -> DiffusionOutput:
        """Convert decoded payload into ``DiffusionOutput``."""

    def predict_noise(self, batch: InputBatch) -> torch.Tensor | None:
        """Run one denoise step and return CFG-merged noise."""

    def advance_scheduler(self, state: DiffusionRequestState, noise: torch.Tensor) -> DiffusionRequestState:
        """Apply one scheduler step and advance ``state.step_index``."""

    def build_model_inputs(self, states: list[DiffusionRequestState]) -> dict[str, Any]:
        """Pack pipeline-private per-request state into batch-local model inputs."""

    def build_step_attention_metadata(self, batch: InputBatch) -> Any:
        """Build attention metadata for one step, if needed."""

    def pack_conditioning(self, state: DiffusionRequestState) -> Any:
        """Pack opaque conditioning for remote stage transport."""

    def unpack_conditioning(self, payload: Any, state: DiffusionRequestState) -> DiffusionRequestState:
        """Unpack opaque conditioning received from another stage."""


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


def supports_diffusion_atoms(pipeline: object) -> bool:
    """Return whether `pipeline` implements the atomized diffusion contract."""

    if not getattr(pipeline, "supports_diffusion_atoms", False):
        return False
    required = (
        "init_state",
        "validation",
        "encoding",
        "preparation",
        "denoising",
        "decoding",
        "postprocess",
        "predict_noise",
        "advance_scheduler",
        "build_model_inputs",
        "build_step_attention_metadata",
    )
    return all(callable(getattr(pipeline, method, None)) for method in required)
