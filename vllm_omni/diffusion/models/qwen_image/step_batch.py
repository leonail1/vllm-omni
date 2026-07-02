# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen-Image step/batch helpers.

The runner deliberately does not know Qwen-private tensor names. Qwen-family
pipelines store those fields in ``DiffusionRequestState.extra`` and use this
module to fold them into ``InputBatch.model_inputs`` for a single DiT step.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Sequence
from typing import Any

import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.stage_kind import DiffusionStageRole, normalize_diffusion_stage_role
from vllm_omni.diffusion.worker.input_batch import InputBatch, set_runner_step_config
from vllm_omni.diffusion.worker.utils import DiffusionRequestState

QWEN_IMAGE_STATE_KEY = "qwen_image"
QWEN_OUTPUT_KEY = "_qwen_image_output"
_QWEN_TRANSPORT_ATOM_ARG_KEYS = frozenset(
    {
        "prompt",
        "negative_prompt",
        "height",
        "width",
        "calculated_height",
        "calculated_width",
        "condition_image_sizes",
        "vae_image_sizes",
        "layers",
        "resolution",
        "cfg_normalize",
        "use_en_prompt",
        "num_inference_steps",
        "sigmas",
        "guidance_scale",
        "num_images_per_prompt",
        "true_cfg_scale",
        "max_sequence_length",
        "output_type",
        "attention_kwargs",
        "callback_on_step_end_tensor_inputs",
    }
)


def qwen_diffusion_stage_role(od_config_or_role: Any) -> DiffusionStageRole:
    role = getattr(od_config_or_role, "diffusion_stage_role", od_config_or_role)
    return normalize_diffusion_stage_role(role)


def qwen_role_loads_scheduler(role: DiffusionStageRole) -> bool:
    # Encoder prepares timesteps and DiT advances them, so both split roles need
    # a scheduler. Decode only consumes final latents.
    return role in (
        DiffusionStageRole.MONOLITHIC,
        DiffusionStageRole.ENCODER,
        DiffusionStageRole.DENOISER,
    )


def qwen_role_loads_text_encoder(role: DiffusionStageRole) -> bool:
    return role in (DiffusionStageRole.MONOLITHIC, DiffusionStageRole.ENCODER)


def qwen_role_loads_transformer(role: DiffusionStageRole) -> bool:
    return role in (DiffusionStageRole.MONOLITHIC, DiffusionStageRole.DENOISER)


def qwen_role_loads_decoder_vae(role: DiffusionStageRole) -> bool:
    return role in (DiffusionStageRole.MONOLITHIC, DiffusionStageRole.DECODER)


def qwen_role_loads_condition_vae(role: DiffusionStageRole) -> bool:
    return role in (DiffusionStageRole.MONOLITHIC, DiffusionStageRole.ENCODER)


def qwen_transformer_in_channels(pipeline: Any) -> int:
    transformer = getattr(pipeline, "transformer", None)
    if transformer is not None:
        return transformer.in_channels
    # Encoder/decoder roles do not instantiate the transformer, but latent
    # preparation still needs the transformer channel count.
    return int(getattr(pipeline, "_qwen_transformer_in_channels", 64))


def qwen_transformer_guidance_embeds(pipeline: Any) -> bool:
    transformer = getattr(pipeline, "transformer", None)
    if transformer is not None:
        return bool(transformer.guidance_embeds)
    return bool(getattr(pipeline, "_qwen_transformer_guidance_embeds", False))


def qwen_vae_metadata(model_path: Any) -> tuple[int, int]:
    config_path = os.path.join(str(model_path), "vae", "config.json")
    try:
        with open(config_path) as f:
            config = json.load(f)
    except OSError:
        return 8, 16

    # Role-specific workers may skip loading VAE weights, but still need the
    # scale factor and latent channel count for shape calculations.
    temporal_downsample = config.get("temporal_downsample") or config.get("temperal_downsample")
    scale_factor = 2 ** len(temporal_downsample) if temporal_downsample else 8
    latent_channels = int(config.get("z_dim", 16))
    return scale_factor, latent_channels


def load_qwen_transformer_weights(pipeline: Any, weights: Any) -> set[str]:
    if getattr(pipeline, "transformer", None) is None:
        # Non-denoiser roles intentionally skip transformer construction and
        # must also ignore transformer weight loading.
        return set()
    from vllm.model_executor.models.utils import AutoWeightsLoader

    loader = AutoWeightsLoader(pipeline)
    return loader.load_weights(weights)


def _drop_qwen_modules(pipeline: Any, names: Sequence[str]) -> list[str]:
    dropped: list[str] = []
    for name in names:
        if hasattr(pipeline, name):
            delattr(pipeline, name)
            dropped.append(name)
    return dropped


def configure_qwen_stage_role(pipeline: Any, role: str) -> list[str]:
    """Release Qwen modules that a split diffusion role will never execute."""

    normalized_role = qwen_diffusion_stage_role(role)
    setattr(pipeline, "_diffusion_stage_role", normalized_role.value)

    if normalized_role == DiffusionStageRole.DENOISER:
        return _drop_qwen_modules(
            pipeline,
            (
                "text_encoder",
                "tokenizer",
                "processor",
                "vl_processor",
                "vae",
                "image_processor",
            ),
        )
    if normalized_role == DiffusionStageRole.DECODER:
        return _drop_qwen_modules(
            pipeline,
            (
                "text_encoder",
                "tokenizer",
                "processor",
                "vl_processor",
                "transformer",
            ),
        )
    return []


def is_qwen_latent_output_type(output_type: Any) -> bool:
    return str(output_type).lower() in {"latent", "latents"}


def qwen_state(state: DiffusionRequestState) -> dict[str, Any]:
    return state.extra.setdefault(QWEN_IMAGE_STATE_KEY, {})


def set_qwen_state(state: DiffusionRequestState, **values: Any) -> None:
    qwen_state(state).update(values)


def get_qwen_state(state: DiffusionRequestState, key: str, default: Any = None) -> Any:
    return qwen_state(state).get(key, default)


def qwen_attention_kwargs(state: DiffusionRequestState) -> dict[str, Any]:
    atom_args = get_qwen_state(state, "atom_args", {})
    if not isinstance(atom_args, dict):
        atom_args = {}
    return get_qwen_state(state, "attention_kwargs", atom_args.get("attention_kwargs")) or {}


def restore_qwen_attention_kwargs(pipeline: Any, state: DiffusionRequestState) -> None:
    setattr(pipeline, "_attention_kwargs", qwen_attention_kwargs(state))


def _clone_qwen_transport_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value
    if isinstance(value, torch.Generator) or callable(value):
        # Generators and callbacks cannot be serialized safely; the runner
        # recreates deterministic generators from sampling.seed instead.
        return None
    if isinstance(value, dict):
        return {
            key: cloned
            for key, item in value.items()
            if (cloned := _clone_qwen_transport_value(item)) is not None
        }
    if isinstance(value, list):
        return [cloned for item in value if (cloned := _clone_qwen_transport_value(item)) is not None]
    if isinstance(value, tuple):
        return tuple(cloned for item in value if (cloned := _clone_qwen_transport_value(item)) is not None)
    return copy.deepcopy(value)


def pack_qwen_conditioning(state: DiffusionRequestState) -> dict[str, Any]:
    """Pack only Qwen state that is safe and needed across stage boundaries."""

    packed: dict[str, Any] = {}
    for key, value in qwen_state(state).items():
        if key == QWEN_OUTPUT_KEY:
            continue
        if key == "atom_args" and isinstance(value, dict):
            # Keep only request semantics needed by later stages. Local objects
            # such as callbacks or generators are intentionally filtered out.
            packed[key] = {
                arg_key: cloned
                for arg_key, arg_value in value.items()
                if arg_key in _QWEN_TRANSPORT_ATOM_ARG_KEYS
                if (cloned := _clone_qwen_transport_value(arg_value)) is not None
            }
            continue
        cloned = _clone_qwen_transport_value(value)
        if cloned is not None:
            packed[key] = cloned
    return packed


def _normalize_prompt_embeds(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        return x
    raise ValueError(f"prompt_embeds must be 2D or 3D, got shape={tuple(x.shape)}")


def _normalize_mask(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim != 2:
        raise ValueError(f"prompt mask must be 1D or 2D, got shape={tuple(x.shape)}")
    if x.dtype != torch.bool:
        x = x != 0
    return x


def _pad_prompt_embeds(x: torch.Tensor, target_seq_len: int) -> torch.Tensor:
    x = _normalize_prompt_embeds(x)
    bsz, seq_len, hidden = x.shape
    if seq_len == target_seq_len:
        return x
    out = x.new_zeros((bsz, target_seq_len, hidden))
    out[:, :seq_len] = x
    return out


def _pad_mask(x: torch.Tensor, target_seq_len: int) -> torch.Tensor:
    x = _normalize_mask(x)
    bsz, seq_len = x.shape
    if seq_len == target_seq_len:
        return x
    out = torch.zeros((bsz, target_seq_len), dtype=torch.bool, device=x.device)
    out[:, :seq_len] = x
    return out


def _prepare_reused_buffer(
    current: torch.Tensor | None,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if current is not None and tuple(current.shape) == shape and current.dtype == dtype and current.device == device:
        return current
    return torch.empty(shape, dtype=dtype, device=device)


def _validate_gather_tensors(
    values: Sequence[torch.Tensor],
    *,
    field_name: str,
) -> tuple[torch.dtype, torch.device, tuple[int, ...], int]:
    if not values:
        raise ValueError(f"Cannot gather empty tensor list for {field_name}.")

    first = values[0]
    dtype = first.dtype
    device = first.device
    suffix_shape = tuple(first.shape[1:])
    total_rows = 0
    for value in values:
        if value.dtype != dtype:
            raise ValueError(f"Mixed dtypes in {field_name} batch.")
        if value.device != device:
            raise ValueError(f"Mixed devices in {field_name} batch.")
        if tuple(value.shape[1:]) != suffix_shape:
            raise ValueError(
                f"Mixed trailing shapes in {field_name} batch: expected {suffix_shape}, got {tuple(value.shape[1:])}."
            )
        total_rows += int(value.shape[0])
    return dtype, device, suffix_shape, total_rows


def _gather_tensor_rows(
    values: Sequence[torch.Tensor],
    *,
    field_name: str,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    dtype, device, suffix_shape, total_rows = _validate_gather_tensors(values, field_name=field_name)
    gathered = _prepare_reused_buffer(
        out,
        shape=(total_rows, *suffix_shape),
        dtype=dtype,
        device=device,
    )

    row_offset = 0
    for value in values:
        next_row_offset = row_offset + int(value.shape[0])
        gathered[row_offset:next_row_offset].copy_(value)
        row_offset = next_row_offset
    return gathered


def _state_tensor(state: DiffusionRequestState, key: str) -> torch.Tensor | None:
    value = get_qwen_state(state, key)
    if value is None:
        return None
    if not torch.is_tensor(value):
        raise ValueError(f"Qwen state field {key!r} must be a tensor.")
    return value


def _prepare_prompt_field_on_state(
    state: DiffusionRequestState,
    *,
    embeds_key: str,
    mask_key: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    embeds = _state_tensor(state, embeds_key)
    if embeds is None:
        return None, None

    normalized_embeds = _normalize_prompt_embeds(embeds)
    if normalized_embeds is not embeds:
        set_qwen_state(state, **{embeds_key: normalized_embeds})
    embeds = normalized_embeds

    mask = _state_tensor(state, mask_key)
    if mask is None:
        return embeds, None

    normalized_mask = _normalize_mask(mask)
    if normalized_mask is not mask:
        set_qwen_state(state, **{mask_key: normalized_mask})
    return embeds, normalized_mask


def _get_seq_lens_from_mask(mask: torch.Tensor) -> list[int]:
    return mask.sum(dim=1, dtype=torch.int32).tolist()


def _get_request_prompt_seq_lens(
    state: DiffusionRequestState,
    *,
    embeds_key: str,
    mask_key: str,
    seq_lens_key: str,
) -> list[int]:
    embeds, mask = _prepare_prompt_field_on_state(
        state,
        embeds_key=embeds_key,
        mask_key=mask_key,
    )
    if embeds is None:
        raise ValueError(f"{embeds_key} is not initialized on request {state.request_id}.")
    if mask is not None:
        if mask.shape[0] != embeds.shape[0]:
            raise ValueError(
                f"{mask_key} batch dimension does not match {embeds_key} for request {state.request_id}."
            )
        return _get_seq_lens_from_mask(mask)
    seq_lens = get_qwen_state(state, seq_lens_key)
    if seq_lens is not None:
        return [int(value) for value in seq_lens]
    return [int(embeds.shape[1])] * int(embeds.shape[0])


def _prepare_request_prompt_field(
    state: DiffusionRequestState,
    *,
    embeds_key: str,
    mask_key: str,
    seq_lens_key: str,
    target_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    embeds, mask = _prepare_prompt_field_on_state(
        state,
        embeds_key=embeds_key,
        mask_key=mask_key,
    )
    if embeds is None:
        raise ValueError(f"{embeds_key} is not initialized on request {state.request_id}.")

    actual_seq_lens = _get_request_prompt_seq_lens(
        state,
        embeds_key=embeds_key,
        mask_key=mask_key,
        seq_lens_key=seq_lens_key,
    )
    max_actual_seq_len = max(actual_seq_lens) if actual_seq_lens else 0

    if mask is None and max_actual_seq_len != target_seq_len:
        raise ValueError(
            f"Variable-length {embeds_key} in batch but {mask_key} is None. "
            f"Provide masks or ensure {embeds_key} have the same seq_len."
        )

    current_seq_len = int(embeds.shape[1])
    if current_seq_len < target_seq_len:
        embeds = _pad_prompt_embeds(embeds, target_seq_len)
        set_qwen_state(state, **{embeds_key: embeds})
        if mask is not None:
            mask = _pad_mask(mask, target_seq_len)
            set_qwen_state(state, **{mask_key: mask})
        current_seq_len = target_seq_len

    if current_seq_len > target_seq_len:
        if max_actual_seq_len > target_seq_len:
            raise ValueError(
                f"{embeds_key} for request {state.request_id} requires seq_len "
                f"{max_actual_seq_len}, got target {target_seq_len}."
            )
        return embeds[:, :target_seq_len], None if mask is None else mask[:, :target_seq_len]

    return embeds, mask


def _prepare_padded_prompt_fields(
    states: Sequence[DiffusionRequestState],
    *,
    embeds_key: str,
    mask_key: str,
    seq_lens_key: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    prepared_fields = [
        _prepare_prompt_field_on_state(
            state,
            embeds_key=embeds_key,
            mask_key=mask_key,
        )
        for state in states
    ]
    embeds_values = [embeds for embeds, _ in prepared_fields]
    if not any(embeds is not None for embeds in embeds_values):
        return None, None
    if not all(embeds is not None for embeds in embeds_values):
        raise ValueError(f"Mixed {embeds_key} in batch.")

    mask_values = [mask for _, mask in prepared_fields]
    if any(mask is None for mask in mask_values) and any(mask is not None for mask in mask_values):
        raise ValueError(f"Mixed {mask_key} in batch.")

    target_seq_len = max(
        max(
            _get_request_prompt_seq_lens(
                state,
                embeds_key=embeds_key,
                mask_key=mask_key,
                seq_lens_key=seq_lens_key,
            )
        )
        for state in states
    )

    # Pad every request to the largest real prompt length in this batch.  The
    # original masks preserve variable-length semantics for Qwen attention.
    request_embeds: list[torch.Tensor] = []
    request_masks: list[torch.Tensor] = []
    for state in states:
        prepared_embeds, prepared_mask = _prepare_request_prompt_field(
            state,
            embeds_key=embeds_key,
            mask_key=mask_key,
            seq_lens_key=seq_lens_key,
            target_seq_len=target_seq_len,
        )
        request_embeds.append(prepared_embeds)
        if prepared_mask is not None:
            request_masks.append(prepared_mask)

    gathered_embeds = _gather_tensor_rows(request_embeds, field_name=embeds_key)
    if not request_masks:
        return gathered_embeds, None
    return gathered_embeds, _gather_tensor_rows(request_masks, field_name=mask_key)


def _prepare_prompt_embeds(states: Sequence[DiffusionRequestState]) -> tuple[torch.Tensor, torch.Tensor | None]:
    prompt_embeds, prompt_embeds_mask = _prepare_padded_prompt_fields(
        states,
        embeds_key="prompt_embeds",
        mask_key="prompt_embeds_mask",
        seq_lens_key="txt_seq_lens",
    )
    if prompt_embeds is None:
        raise ValueError("All requests must have Qwen `prompt_embeds` initialized.")
    return prompt_embeds, prompt_embeds_mask


def _prepare_negative_prompt_embeds(
    states: Sequence[DiffusionRequestState],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    return _prepare_padded_prompt_fields(
        states,
        embeds_key="negative_prompt_embeds",
        mask_key="negative_prompt_embeds_mask",
        seq_lens_key="negative_txt_seq_lens",
    )


def _prepare_optional_tensor(
    states: Sequence[DiffusionRequestState],
    key: str,
) -> torch.Tensor | None:
    values = [_state_tensor(state, key) for state in states]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"Mixed {key} presence in one Qwen batch is not supported.")
    return _gather_tensor_rows([value for value in values if value is not None], field_name=key)


def _prepare_seq_lens(states: Sequence[DiffusionRequestState], key: str) -> list[int] | None:
    values = [get_qwen_state(state, key) for state in states]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"Mixed {key} in batch.")
    return [int(item) for value in values if value is not None for item in value]


def _prepare_img_shapes(states: Sequence[DiffusionRequestState]) -> list | None:
    values = [get_qwen_state(state, "img_shapes") for state in states]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("Mixed img_shapes in batch.")
    return [item for value in values if value is not None for item in value]


def _attention_kwargs_equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        result = left == right
    except Exception:
        return False
    return result if isinstance(result, bool) else False


def _prepare_attention_kwargs(states: Sequence[DiffusionRequestState]) -> dict[str, Any]:
    values = [qwen_attention_kwargs(state) for state in states]
    first = values[0]
    if any(not _attention_kwargs_equal(first, value) for value in values[1:]):
        raise ValueError("Mixed attention_kwargs in one Qwen batch are not supported.")
    return first


def build_qwen_model_inputs(states: Sequence[DiffusionRequestState]) -> dict[str, Any]:
    prompt_embeds, prompt_embeds_mask = _prepare_prompt_embeds(states)
    negative_prompt_embeds, negative_prompt_embeds_mask = _prepare_negative_prompt_embeds(states)
    additional_t_cond = _prepare_optional_tensor(states, "additional_t_cond")
    # Attention kwargs are request-scoped Qwen state. Carry them through the
    # batch input instead of reading a mutable pipeline attribute during denoise.
    extra_transformer_kwargs = {"attention_kwargs": _prepare_attention_kwargs(states)}
    if additional_t_cond is not None:
        extra_transformer_kwargs["additional_t_cond"] = additional_t_cond
    return {
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": prompt_embeds_mask,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
        "guidance": _prepare_optional_tensor(states, "guidance"),
        "image_latents": _prepare_optional_tensor(states, "image_latents"),
        "img_shapes": _prepare_img_shapes(states),
        "txt_seq_lens": _prepare_seq_lens(states, "txt_seq_lens"),
        "negative_txt_seq_lens": _prepare_seq_lens(states, "negative_txt_seq_lens"),
        "extra_transformer_kwargs": extra_transformer_kwargs,
    }



class QwenImageStepAtomsMixin:
    """Shared atom defaults for Qwen-Image family step/batch execution."""

    supports_diffusion_atoms = True
    supports_varlen_batch = True
    supports_stage_split = True

    def configure_diffusion_stage_role(self, role: str) -> None:
        configure_qwen_stage_role(self, role)

    def init_state(self, req: OmniDiffusionRequest) -> DiffusionRequestState:
        return DiffusionRequestState(
            request_id=req.request_id,
            sampling=copy.deepcopy(req.sampling_params),
            prompts=req.prompts,
        )

    def build_model_inputs(self, states: list[DiffusionRequestState]) -> dict[str, Any]:
        return build_qwen_model_inputs(states)

    def build_step_attention_metadata(self, batch: InputBatch) -> Any:
        del batch
        return {}

    def _build_denoise_kwargs(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor | None,
        img_shapes: list,
        txt_seq_lens: list[int] | None,
        do_true_cfg: bool,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds_mask: torch.Tensor | None,
        negative_txt_seq_lens: list[int] | None,
        image_latents: torch.Tensor | None = None,
        extra_transformer_kwargs: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, int | None]:
        extra_transformer_kwargs = extra_transformer_kwargs or {}
        t_for_model = timestep.expand(latents.shape[0]).to(
            device=latents.device,
            dtype=latents.dtype,
        )

        latent_model_input = latents
        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1)

        positive_kwargs = {
            "hidden_states": latent_model_input,
            "timestep": t_for_model / 1000,
            "guidance": guidance,
            "encoder_hidden_states_mask": prompt_embeds_mask,
            "encoder_hidden_states": prompt_embeds,
            "img_shapes": img_shapes,
            "txt_seq_lens": txt_seq_lens,
            **extra_transformer_kwargs,
        }
        if do_true_cfg:
            negative_kwargs = {
                "hidden_states": latent_model_input,
                "timestep": t_for_model / 1000,
                "guidance": guidance,
                "encoder_hidden_states_mask": negative_prompt_embeds_mask,
                "encoder_hidden_states": negative_prompt_embeds,
                "img_shapes": img_shapes,
                "txt_seq_lens": negative_txt_seq_lens,
                **extra_transformer_kwargs,
            }
        else:
            negative_kwargs = None

        output_slice = latents.size(1) if image_latents is not None else None
        return positive_kwargs, negative_kwargs, output_slice

    def predict_noise(
        self,
        input_batch: InputBatch | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        if not isinstance(input_batch, InputBatch):
            if input_batch is None:
                return super().predict_noise(*args, **kwargs)
            return super().predict_noise(input_batch, *args, **kwargs)

        del args, kwargs
        if self.interrupt:
            return None

        t = input_batch.timesteps
        model_inputs = input_batch.model_inputs
        self._current_timestep = t
        self.transformer.do_true_cfg = input_batch.do_true_cfg

        extra_transformer_kwargs = {
            "attention_kwargs": self.attention_kwargs,
            "return_dict": False,
        }
        extra_transformer_kwargs.update(model_inputs.get("extra_transformer_kwargs", {}))
        positive_kwargs, negative_kwargs, output_slice = self._build_denoise_kwargs(
            latents=input_batch.latents,
            timestep=t,
            guidance=model_inputs.get("guidance"),
            prompt_embeds=model_inputs["prompt_embeds"],
            prompt_embeds_mask=model_inputs.get("prompt_embeds_mask"),
            img_shapes=model_inputs.get("img_shapes"),
            txt_seq_lens=model_inputs.get("txt_seq_lens"),
            do_true_cfg=input_batch.do_true_cfg,
            negative_prompt_embeds=model_inputs.get("negative_prompt_embeds"),
            negative_prompt_embeds_mask=model_inputs.get("negative_prompt_embeds_mask"),
            negative_txt_seq_lens=model_inputs.get("negative_txt_seq_lens"),
            image_latents=model_inputs.get("image_latents"),
            extra_transformer_kwargs=extra_transformer_kwargs,
        )
        return self.predict_noise_maybe_with_cfg(
            input_batch.do_true_cfg,
            input_batch.true_cfg_scale,
            positive_kwargs,
            negative_kwargs,
            input_batch.cfg_normalize,
            output_slice,
        )

    def advance_scheduler(
        self,
        state: DiffusionRequestState,
        noise_pred: torch.Tensor,
        **kwargs: Any,
    ) -> DiffusionRequestState:
        del kwargs
        if self.interrupt:
            return state
        state.latents = self.scheduler_step_maybe_with_cfg(
            noise_pred,
            state.current_timestep,
            state.latents,
            bool(get_qwen_state(state, "do_true_cfg", False)),
            per_request_scheduler=state.scheduler,
        )
        state.step_index += 1
        return state

    def denoising(self, state: DiffusionRequestState) -> DiffusionRequestState:
        while not state.denoise_completed:
            input_batch = InputBatch.make_batch([state])
            input_batch.model_inputs.update(self.build_model_inputs(input_batch.states))
            noise_pred = self.predict_noise(input_batch)
            if noise_pred is None:
                break
            self.advance_scheduler(state, noise_pred)
        return state

    def _decode_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        output_type: str = "pil",
    ) -> DiffusionOutput:
        if is_qwen_latent_output_type(output_type):
            return DiffusionOutput(
                output=latents,
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        return DiffusionOutput(
            output=image,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def decoding(self, state: DiffusionRequestState) -> DiffusionRequestState:
        self._current_timestep = None
        args = get_qwen_state(state, "atom_args", {})
        height = args.get("height", state.sampling.height or self.default_sample_size * self.vae_scale_factor)
        width = args.get("width", state.sampling.width or self.default_sample_size * self.vae_scale_factor)
        output_type = get_qwen_state(state, "output_type", args.get("output_type", "pil"))
        set_qwen_state(state, **{QWEN_OUTPUT_KEY: self._decode_latents(state.latents, height, width, output_type)})
        return state

    def postprocess(self, state: DiffusionRequestState) -> DiffusionOutput:
        output = get_qwen_state(state, QWEN_OUTPUT_KEY)
        if output is None:
            raise ValueError(f"Request {state.request_id} has not been decoded.")
        return output

    def pack_conditioning(self, state: DiffusionRequestState) -> Any:
        return pack_qwen_conditioning(state)

    def unpack_conditioning(self, payload: Any, state: DiffusionRequestState) -> DiffusionRequestState:
        if isinstance(payload, dict):
            set_qwen_state(state, **payload)
        else:
            state.conditioning = payload
        return state

    def rehydrate_stage_state(self, state: DiffusionRequestState) -> DiffusionRequestState:
        if state.sampling.output_type is not None:
            set_qwen_state(state, output_type=state.sampling.output_type)
        restore_qwen_attention_kwargs(self, state)
        if state.scheduler is None and getattr(self, "scheduler", None) is not None:
            # Rebuild per-request scheduler state on the receiving role because
            # scheduler objects are deliberately not part of the transport payload.
            prepare_timesteps = getattr(self, "prepare_timesteps", None)
            if callable(prepare_timesteps) and state.latents is not None:
                num_steps = state.sampling.num_inference_steps or state.total_steps
                timesteps, _ = prepare_timesteps(
                    num_steps,
                    state.sampling.sigmas,
                    state.latents.shape[1],
                )
                if state.timesteps is None:
                    state.timesteps = timesteps
                state.scheduler = copy.deepcopy(self.scheduler)
            else:
                state.scheduler = copy.deepcopy(self.scheduler)
            set_begin_index = getattr(state.scheduler, "set_begin_index", None)
            if callable(set_begin_index):
                set_begin_index(state.step_index)
        return state

    def _set_runner_cfg(
        self,
        state: DiffusionRequestState,
        *,
        do_true_cfg: bool,
        true_cfg_scale: float,
        cfg_normalize: bool = True,
    ) -> None:
        set_runner_step_config(
            state,
            do_true_cfg=do_true_cfg,
            true_cfg_scale=true_cfg_scale,
            cfg_normalize=cfg_normalize,
        )
