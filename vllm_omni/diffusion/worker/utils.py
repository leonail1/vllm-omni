# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request mutable state for step-wise diffusion execution."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.inputs.data import OmniPromptType


TransportBoundary = Literal["encode_to_dit", "dit_to_decode"]
_SAMPLING_TENSOR_FIELDS = (
    "latents",
    "audio_latents",
    "raw_latent_shape",
    "noise_pred",
    "image_latent",
    "output",
    "timesteps",
    "timestep",
    "trajectory_timesteps",
    "trajectory_latents",
)


@dataclass
class DiffusionStateTransport:
    """Plain transport payload for stage-split diffusion state.

    Tensor movement can be backed by msgpack tensor serialization or by a
    TensorBuffer.  The model state only exposes the schema: non-tensor metadata
    and named tensor fields.
    """

    boundary: TransportBoundary
    request_id: str
    sampling: OmniDiffusionSamplingParams
    prompts: list[OmniPromptType] | None
    meta: dict[str, Any] = field(default_factory=dict)
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def _clone_sampling_for_transport(sampling: "OmniDiffusionSamplingParams") -> "OmniDiffusionSamplingParams":
    cloned = copy.deepcopy(sampling)
    cloned.generator = None
    for name in _SAMPLING_TENSOR_FIELDS:
        setattr(cloned, name, None)
    return cloned


def _sampling_from_transport(value: OmniDiffusionSamplingParams | dict[str, Any]) -> OmniDiffusionSamplingParams:
    if isinstance(value, OmniDiffusionSamplingParams):
        return value
    return OmniDiffusionSamplingParams(**value)


def _contains_tensor(value: Any) -> bool:
    if torch.is_tensor(value):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def _move_tensor_tree(value: Any, device: torch.device | str | None) -> Any:
    if device is None:
        return value
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_tensor_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensor_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensor_tree(item, device) for item in value)
    return value


@dataclass
class DiffusionRequestState:
    """Per-request mutable state across all pipeline stages.

    Owned by Runner and passed through all step-execution stages:
    ``encode_stage()`` initializes/updates fields, ``denoise_stage()`` and
    ``scheduler_stage()`` mutate per-step fields, and ``decode_stage()``
    consumes final latents. This state object is also the cache unit for
    future continuous batching.

    This dataclass keeps only the minimal cross-model state required by the
    step-execution contract. Pipeline-specific state should be stored in
    ``extra`` and promoted here only when it becomes shared across models.

    Examples:
    - Wan-style pipelines may keep ``condition``, ``first_frame_mask``, or
      ``image_embeds`` in ``extra``.
    - Bagel-style pipelines may keep ``gen_context``,
      ``cfg_text_context``, ``cfg_img_context``, or ``image_shape`` in
      ``extra``.
    """

    # ── Identity / request-level inputs ──
    request_id: str
    sampling: OmniDiffusionSamplingParams
    prompts: list[OmniPromptType] | None = None

    # ── Encoded prompts (set once by encode_stage) ──
    prompt_embeds: torch.Tensor | None = None
    prompt_embeds_mask: torch.Tensor | None = None
    negative_prompt_embeds: torch.Tensor | None = None
    negative_prompt_embeds_mask: torch.Tensor | None = None

    # ── Latent state (mutated every step by scheduler_stage) ──
    latents: torch.Tensor | None = None

    # ── Timestep schedule (set once by encode_stage) ──
    timesteps: torch.Tensor | list[torch.Tensor] | None = None
    step_index: int = 0
    chunk_index: int = 0
    step_in_chunk: int = 0
    total_chunks: int = 1
    chunk_num_steps: int | None = None

    # ── Per-request scheduler instance (set once by encode_stage) ──
    scheduler: Any | None = None

    # ── CFG config (set once by encode_stage) ──
    do_true_cfg: bool = False
    guidance: torch.Tensor | None = None

    # ── Spatial / sequence metadata (set once by encode_stage) ──
    img_shapes: list | None = None
    txt_seq_lens: list[int] | None = None
    negative_txt_seq_lens: list[int] | None = None

    # Pipeline-specific extras. Keep model-private fields here unless they
    # become part of the shared step-execution contract.
    # For example: Wan condition tensors / masks, or Bagel KV contexts.
    extra: dict[str, Any] = field(default_factory=dict)

    # ── Properties ──

    @property
    def current_timestep(self) -> torch.Tensor | None:
        if self.timesteps is None:
            return None
        if self.step_index >= self.total_steps:
            return None
        if isinstance(self.timesteps, torch.Tensor):
            if self.timesteps.ndim == 0:
                return self.timesteps
            return self.timesteps[self.step_index]
        return self.timesteps[self.step_index]

    @property
    def total_steps(self) -> int:
        if self.timesteps is None:
            return 0
        if isinstance(self.timesteps, torch.Tensor):
            if self.timesteps.ndim == 0:
                return 1
            return int(self.timesteps.shape[0])
        return len(self.timesteps)

    @property
    def denoise_completed(self) -> bool:
        total_steps = self.total_steps
        if total_steps == 0:
            return False
        return self.step_index >= total_steps

    @property
    def new_request(self) -> bool:
        # TODO: this is only an approximation for current stepwise mode.
        # A real "new request" signal should eventually come from scheduler/runner state transitions.
        return self.step_index == 0 or self.timesteps is None

    def to_transport(
        self,
        boundary: TransportBoundary,
    ) -> DiffusionStateTransport:
        """Build a plain payload for a stage boundary.

        RunnerV2 owns the IPC/TensorBuffer mechanics.
        """
        # TODO: Move these hard-coded field lists into a model-declared
        # transport contract after qwen-image-edit/Wan migrations clarify it.
        tensors: dict[str, torch.Tensor] = {}
        tensor_names = ("latents",)
        if boundary == "encode_to_dit":
            tensor_names = (
                "prompt_embeds",
                "prompt_embeds_mask",
                "negative_prompt_embeds",
                "negative_prompt_embeds_mask",
                "latents",
                "guidance",
            )

        for name in tensor_names:
            value = getattr(self, name)
            if torch.is_tensor(value):
                tensors[name] = value

        if boundary == "encode_to_dit" and torch.is_tensor(self.timesteps):
            tensors["timesteps"] = self.timesteps

        for name in _SAMPLING_TENSOR_FIELDS:
            value = getattr(self.sampling, name, None)
            if _contains_tensor(value):
                tensors[f"sampling.{name}"] = value

        meta = {
            "step_index": self.step_index,
            "chunk_index": self.chunk_index,
            "step_in_chunk": self.step_in_chunk,
            "total_chunks": self.total_chunks,
            "chunk_num_steps": self.chunk_num_steps,
            "do_true_cfg": self.do_true_cfg,
            "img_shapes": self.img_shapes,
            "txt_seq_lens": self.txt_seq_lens,
            "negative_txt_seq_lens": self.negative_txt_seq_lens,
            "timesteps_is_tensor": torch.is_tensor(self.timesteps),
            "timesteps_value": None if torch.is_tensor(self.timesteps) else self.timesteps,
        }
        extra = copy.deepcopy(self.extra)
        if boundary == "dit_to_decode":
            extra = {}
            meta = {
                "step_index": self.step_index,
                "chunk_index": self.chunk_index,
                "step_in_chunk": self.step_in_chunk,
                "total_chunks": self.total_chunks,
                "chunk_num_steps": self.chunk_num_steps,
            }

        return DiffusionStateTransport(
            boundary=boundary,
            request_id=self.request_id,
            sampling=_clone_sampling_for_transport(self.sampling),
            prompts=self.prompts,
            meta=meta,
            tensors=tensors,
            extra=extra,
        )

    @classmethod
    def from_transport(
        cls,
        payload: DiffusionStateTransport | dict[str, Any],
        *,
        device: torch.device | str | None = None,
    ) -> "DiffusionRequestState":
        if isinstance(payload, dict):
            payload = DiffusionStateTransport(**payload)
        tensors = payload.tensors

        def _tensor(name: str) -> torch.Tensor | None:
            # Move tensors at the receiving role boundary, not during serialization.
            value = tensors.get(name)
            if value is not None and device is not None:
                return value.to(device)
            return value

        state = cls(
            request_id=payload.request_id,
            sampling=_sampling_from_transport(payload.sampling),
            prompts=payload.prompts,
        )
        meta = payload.meta
        state.prompt_embeds = _tensor("prompt_embeds")
        state.prompt_embeds_mask = _tensor("prompt_embeds_mask")
        state.negative_prompt_embeds = _tensor("negative_prompt_embeds")
        state.negative_prompt_embeds_mask = _tensor("negative_prompt_embeds_mask")
        state.latents = _tensor("latents")
        state.timesteps = _tensor("timesteps") if meta.get("timesteps_is_tensor") else meta.get("timesteps_value")
        state.guidance = _tensor("guidance")
        state.step_index = int(meta.get("step_index", 0))
        state.chunk_index = int(meta.get("chunk_index", 0))
        state.step_in_chunk = int(meta.get("step_in_chunk", 0))
        state.total_chunks = int(meta.get("total_chunks", 1))
        state.chunk_num_steps = meta.get("chunk_num_steps")
        state.do_true_cfg = bool(meta.get("do_true_cfg", False))
        state.img_shapes = meta.get("img_shapes")
        state.txt_seq_lens = meta.get("txt_seq_lens")
        state.negative_txt_seq_lens = meta.get("negative_txt_seq_lens")
        for name in _SAMPLING_TENSOR_FIELDS:
            value = tensors.get(f"sampling.{name}")
            if value is not None:
                setattr(state.sampling, name, _move_tensor_tree(value, device))

        state.extra = copy.deepcopy(payload.extra)
        return state


class BaseRunnerOutput(ABC):
    @abstractmethod
    def get_request_output(self, request_id: str) -> RunnerOutput | None:
        pass


@dataclass
class RunnerOutput(BaseRunnerOutput):
    """Output of a single denoising step for a request.

    NOTE: `latents` may be None when returned through IPC to avoid
    serialization overhead. The actual latents are kept in Worker's
    _request_state_cache.
    """

    request_id: str
    step_index: int | None = None
    finished: bool = False
    result: DiffusionOutput | None = None

    def get_request_output(self, request_id: str) -> RunnerOutput | None:
        return self if self.request_id == request_id else None


@dataclass
class BatchRunnerOutput(BaseRunnerOutput):
    runner_outputs: list[RunnerOutput]
    _id_to_idx: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._id_to_idx = {out.request_id: i for i, out in enumerate(self.runner_outputs)}

    def __getitem__(self, request_id: str) -> RunnerOutput | None:
        """access single RunnerOutput by request_id"""
        idx = self._id_to_idx.get(request_id)
        return self.runner_outputs[idx] if idx is not None else None

    def get_request_output(self, request_id: str) -> RunnerOutput | None:
        return self[request_id]

    @property
    def request_ids(self) -> list[str]:
        return list(self._id_to_idx.keys())

    def __len__(self) -> int:
        return len(self.runner_outputs)

    @classmethod
    def from_list(cls, runner_output_list: list[RunnerOutput]) -> BatchRunnerOutput:
        return cls(runner_outputs=runner_output_list)
