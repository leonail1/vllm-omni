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

    from vllm_omni.data_entry_keys import OmniPayload
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
class DiffusionStageAtoms(Protocol):
    """Request-level diffusion atoms shared by request mode and step mode."""

    supports_step_execution: ClassVar[bool] = True

    def init_state(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Initialize pipeline-private fields on a newly created request state."""
        ...

    def check_inputs(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Validate request inputs before model work begins."""
        ...

    def encode(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Run text/input encoders and populate encoded prompt fields."""
        ...

    def prepare(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Prepare model-specific denoise state after encode."""
        ...

    def diffuse(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Run the full diffusion loop for request-mode/golden-path execution."""
        ...

    def decode(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Decode raw latent state into the model output representation."""
        ...

    def postprocess(self, state: DiffusionRequestState) -> DiffusionOutput:
        """Apply model-specific output post-processing and return final output."""
        ...


@runtime_checkable
class DiffusionStepAtoms(Protocol):
    """Single-step atoms used by continuous batching in runner v2."""

    def build_model_inputs(
        self,
        states: list[DiffusionRequestState],
    ) -> dict[str, Any]:
        """Build model-specific batched inputs before attention metadata."""
        ...

    def build_step_attention_metadata(
        self,
        input_batch: InputBatch,
    ) -> Any:
        """Build step-local attention metadata for the current input batch."""
        ...

    def denoise_step(self, input_batch: InputBatch, **kwargs: Any) -> torch.Tensor | None:
        """Run one DiT denoise step on the runner-assembled batch."""
        ...

    def step_scheduler(
        self,
        state: DiffusionRequestState,
        noise_pred: torch.Tensor,
    ) -> DiffusionRequestState:
        """Apply one scheduler step to a request-local state."""
        ...


class DefaultDiffusionStepAtomsMixin:
    """Default no-op implementations for optional step atom hooks."""

    def build_model_inputs(
        self,
        states: list[DiffusionRequestState],
    ) -> dict[str, Any]:
        """Return extra model-private batched inputs for ``InputBatch``."""
        return {}

    def build_step_attention_metadata(
        self,
        input_batch: InputBatch,
    ) -> Any:
        """Return runner-level attention metadata for this step batch."""
        return None


@runtime_checkable
class SupportsStepExecution(Protocol):
    """Backward-compatible marker for pipelines that opt into step execution."""

    supports_step_execution: ClassVar[bool] = True


SupportsStepAtoms = DiffusionStepAtoms


@runtime_checkable
class SupportsStepPayloadCodec(Protocol):
    """Pack/unpack fine-grained diffusion stage payloads.

    Implementations should use the existing ``OmniPayload`` dictionary shape
    and connector mixin caches instead of introducing a diffusion-specific
    transport framework.
    """

    def pack_encode_payload(self, state: DiffusionRequestState) -> OmniPayload:
        """Pack encoder outputs needed by the DiT stage."""
        ...

    def unpack_encode_payload(
        self,
        payload: OmniPayload,
        state: DiffusionRequestState,
    ) -> None:
        """Unpack encoder-stage payload into runner-owned request state."""
        ...

    def pack_decode_payload(self, state: DiffusionRequestState) -> OmniPayload:
        """Pack DiT outputs needed by the decoder stage."""
        ...

    def unpack_decode_payload(
        self,
        payload: OmniPayload,
        state: DiffusionRequestState,
    ) -> None:
        """Unpack DiT-stage payload before decode."""
        ...

    def pack_output_payload(
        self,
        output: DiffusionOutput,
        request_id: str,
    ) -> OmniPayload:
        """Pack decoder output for downstream/local delivery."""
        ...

    def unpack_output_payload(self, payload: OmniPayload) -> DiffusionOutput:
        """Unpack decoder output payload."""
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
    """Return whether `pipeline` opts into step execution."""

    return getattr(pipeline, "supports_step_execution", False) is True


def supports_stage_atoms(pipeline: object) -> bool:
    """Return whether `pipeline` implements request-level diffusion atoms."""

    return supports_step_execution(pipeline) and isinstance(pipeline, DiffusionStageAtoms)


def supports_step_atoms(pipeline: object) -> bool:
    """Return whether `pipeline` implements single-step diffusion atoms."""

    return isinstance(pipeline, DiffusionStepAtoms)


def supports_step_payload_codec(pipeline: object) -> bool:
    """Return whether `pipeline` can pack/unpack step stage payloads."""

    return isinstance(pipeline, SupportsStepPayloadCodec)
