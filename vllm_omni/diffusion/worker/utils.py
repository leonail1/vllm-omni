# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request mutable state for step-wise diffusion execution."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType


TransportBoundary = Literal["encode_to_dit", "dit_to_decode"]
_SAMPLING_TENSOR_FIELDS = (
    "latents",
    "audio_latents",
    "raw_latent_shape",
    "noise_pred",
    "image_latent",
    "timesteps",
    "timestep",
    "trajectory_timesteps",
    "trajectory_latents",
)
_RUNNER_STEP_EXTRA_KEY = "_runner_step"
_CONDITIONING_UNSET = object()


@dataclass
class DiffusionStateTransport:
    """Plain payload for diffusion stage-split state transport."""

    boundary: TransportBoundary
    request_id: str
    sampling: OmniDiffusionSamplingParams
    prompts: list[OmniPromptType] | None
    meta: dict[str, Any] = field(default_factory=dict)
    tensors: dict[str, Any] = field(default_factory=dict)
    conditioning: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _clone_sampling_for_transport(sampling: "OmniDiffusionSamplingParams") -> "OmniDiffusionSamplingParams":
    sampling_for_copy = copy.copy(sampling)
    if hasattr(sampling_for_copy, "generator"):
        sampling_for_copy.generator = None
    cloned = copy.deepcopy(sampling_for_copy)
    if hasattr(cloned, "generator"):
        cloned.generator = None
    for name in _SAMPLING_TENSOR_FIELDS:
        if hasattr(cloned, name):
            setattr(cloned, name, None)
    return cloned


def _sampling_from_transport(value: OmniDiffusionSamplingParams | dict[str, Any]) -> OmniDiffusionSamplingParams:
    if not isinstance(value, dict):
        return value
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

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


def _sanitize_prompt_for_transport(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if torch.is_tensor(value) or isinstance(value, torch.Generator) or callable(value):
        return None
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"multi_modal_data", "additional_information"}:
                continue
            cleaned = _sanitize_prompt_for_transport(item)
            if cleaned is not None:
                sanitized[key] = cleaned
        return sanitized
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := _sanitize_prompt_for_transport(item)) is not None]
    if isinstance(value, tuple):
        return tuple(cleaned for item in value if (cleaned := _sanitize_prompt_for_transport(item)) is not None)
    return None


def _sanitize_prompts_for_transport(prompts: list[OmniPromptType] | None) -> list[OmniPromptType] | None:
    if prompts is None:
        return None
    return [_sanitize_prompt_for_transport(prompt) for prompt in prompts]


@dataclass
class DiffusionRequestState:
    """Per-request mutable state across all pipeline stages.

    Owned by Runner and passed through all step-execution stages:
    ``validation()`` / ``encoding()`` / ``preparation()`` initialize request
    inputs, ``predict_noise()`` / ``advance_scheduler()`` mutate per-step
    fields, and ``decoding()`` / ``postprocess()`` consume final latents. This
    state object is also the cache unit for continuous batching.

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

    # ── Latent state (mutated every step by advance_scheduler) ──
    latents: torch.Tensor | None = None

    # ── Timestep schedule (set once by preparation) ──
    timesteps: torch.Tensor | list[torch.Tensor] | None = None
    step_index: int = 0

    # ── Per-request scheduler instance (set once by preparation) ──
    scheduler: Any | None = None

    # Optional typed/opaque model-private conditioning payload. Pipelines may
    # use this instead of ``extra`` when a local typed object is convenient.
    conditioning: Any | None = None

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
        *,
        conditioning: Any = _CONDITIONING_UNSET,
    ) -> DiffusionStateTransport:
        tensors: dict[str, torch.Tensor] = {}
        if torch.is_tensor(self.latents):
            tensors["latents"] = self.latents
        timesteps_in_tensors = boundary == "encode_to_dit" and _contains_tensor(self.timesteps)
        if timesteps_in_tensors:
            tensors["timesteps"] = self.timesteps
        for name in _SAMPLING_TENSOR_FIELDS:
            value = getattr(self.sampling, name, None)
            if _contains_tensor(value):
                tensors[f"sampling.{name}"] = value

        model_conditioning = self.conditioning if conditioning is _CONDITIONING_UNSET else conditioning
        runner_step_config = self.extra.get(_RUNNER_STEP_EXTRA_KEY)
        # Encode->DiT carries model-private conditioning. Once DiT finishes,
        # decode only needs public state such as final latents and sampling.
        extra = copy.deepcopy(self.extra) if model_conditioning is None and boundary == "encode_to_dit" else {}
        return DiffusionStateTransport(
            boundary=boundary,
            request_id=self.request_id,
            sampling=_clone_sampling_for_transport(self.sampling),
            prompts=_sanitize_prompts_for_transport(self.prompts),
            meta={
                "step_index": self.step_index,
                "timesteps_is_tensor": timesteps_in_tensors,
                "timesteps_value": None if timesteps_in_tensors else self.timesteps,
                "runner_step_config": copy.deepcopy(runner_step_config),
            },
            tensors=tensors,
            conditioning=copy.deepcopy(model_conditioning),
            extra=extra,
        )

    @classmethod
    def from_transport(
        cls,
        payload: DiffusionStateTransport | dict[str, Any],
        *,
        device: torch.device | str | None = None,
    ) -> DiffusionRequestState:
        if isinstance(payload, dict):
            payload = DiffusionStateTransport(**payload)
        tensors = payload.tensors

        def _tensor(name: str) -> torch.Tensor | None:
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
        state.latents = _tensor("latents")
        # Timesteps may be tensor-valued on NPU/GPU and must keep device
        # placement, but simple Python schedules are cheaper to keep in meta.
        state.timesteps = (
            _move_tensor_tree(tensors.get("timesteps"), device)
            if meta.get("timesteps_is_tensor")
            else _move_tensor_tree(meta.get("timesteps_value"), device)
        )
        state.step_index = int(meta.get("step_index", 0))
        state.conditioning = _move_tensor_tree(payload.conditioning, device)
        state.extra = _move_tensor_tree(copy.deepcopy(payload.extra), device)
        runner_step_config = meta.get("runner_step_config")
        if runner_step_config is not None:
            state.extra[_RUNNER_STEP_EXTRA_KEY] = copy.deepcopy(runner_step_config)
        for name in _SAMPLING_TENSOR_FIELDS:
            value = tensors.get(f"sampling.{name}")
            if value is not None:
                setattr(state.sampling, name, _move_tensor_tree(value, device))
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
