# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import inspect
import json
import logging
import math
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
import torch.distributed
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import DistributedAutoencoderKLQwenImage
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.dmd2 import DMD2PipelineMixin
from vllm_omni.diffusion.models.composed_pipeline import ComposedDiffusionPipeline
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.qwen_image.cfg_parallel import (
    QwenImageCFGParallelMixin,
)
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageTransformer2DModel,
)
from vllm_omni.diffusion.models.qwen_image.rope_utils import txt_seq_lens_from_embeds
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.prompt_utils import (
    validate_prompt_sequence_lengths,
)
from vllm_omni.diffusion.utils.size_utils import (
    normalize_min_aligned_size,
)
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import DiffusionRequestState

from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

logger = logging.getLogger(__name__)


def get_qwen_image_post_process_func(
    od_config: OmniDiffusionConfig,
):
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])
    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** len(vae_config["temporal_downsample"]) if "temporal_downsample" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    def post_process_func(
        images: torch.Tensor,
    ):
        return image_processor.postprocess(images)

    return post_process_func


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, int]:
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: torch.Tensor | tuple[torch.Tensor],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class QwenImagePipeline(
    ComposedDiffusionPipeline,
    nn.Module,
    QwenImageCFGParallelMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    ENCODE_ATOMS: ClassVar[tuple[str, ...]] = (
        "check_inputs",
        "encode_prompt",
        # Qwen timestep preparation needs the packed latent sequence length.
        "prepare_latents",
        "prepare_timesteps",
    )
    DEFAULT_VAE_SCALE_FACTOR: ClassVar[int] = 8
    _ROLE_COMPONENTS: ClassVar[dict[str, set[str]]] = {
        "all": {"scheduler", "text_encoder", "tokenizer", "vae", "transformer"},
        "encode": {"scheduler", "text_encoder", "tokenizer"},
        "dit": {"scheduler", "transformer"},
        "decode": {"vae"},
    }

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        # Initialize only the components owned by this stage role.
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.device = get_local_device()
        model = od_config.model
        self.stage_role = self._resolve_stage_role(od_config)
        owned_components = self._ROLE_COMPONENTS[self.stage_role]
        owns_transformer = "transformer" in owned_components

        # Role ownership controls both component loading and discovery.
        self._dit_modules = ["transformer"] if owns_transformer else []
        self._encoder_modules = ["text_encoder"] if "text_encoder" in owned_components else []
        self._vae_modules = ["vae"] if "vae" in owned_components else []
        self.weights_sources = (
            [
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=od_config.model,
                    subfolder="transformer",
                    revision=None,
                    prefix="transformer.",
                    fall_back_to_pt=True,
                )
            ]
            if owns_transformer
            else []
        )
        self.weights_loaded_by_model_init = not owns_transformer
        # Check if model is a local path
        local_files_only = os.path.isdir(model)

        # See pipeline_qwen_image_edit_plus: guard against transformers v5
        # multi-worker race on partial subfolder shard sets (Buildkite #1043).
        qwen_subfolders = [
            subfolder for subfolder in ("scheduler", "text_encoder", "vae", "tokenizer") if subfolder in owned_components
        ]
        prefetch_subfolders(
            model,
            qwen_subfolders,
            local_files_only=local_files_only,
        )

        self.scheduler = (
            FlowMatchEulerDiscreteScheduler.from_pretrained(
                model, subfolder="scheduler", local_files_only=local_files_only
            )
            if "scheduler" in owned_components
            else None
        )
        # ``from_pretrained_with_prefetch`` re-prefetches and retries on a
        # half-written cache (missing-shard ``OSError`` *and* the default
        # -config size-mismatch ``RuntimeError`` that ``retry_on_missing_shard``
        # could not recover) instead of crashing the worker.
        self.text_encoder = (
            from_pretrained_with_prefetch(
                Qwen2_5_VLForConditionalGeneration.from_pretrained,
                model,
                subfolder="text_encoder",
                prefetch_list=qwen_subfolders,
                local_files_only=local_files_only,
            )
            if "text_encoder" in owned_components
            else None
        )
        # Qwen2.5-VL ships a vision tower that text-to-image does not use.
        # Drop it while the model is still on CPU, before moving to GPU, so
        # the vision tower never consumes GPU memory. Handle both transformers
        # layouts: newer puts visual under .model, older puts it directly on
        # the model.
        if self.text_encoder is not None:
            visual_owner = None
            if hasattr(self.text_encoder, "model") and hasattr(self.text_encoder.model, "visual"):
                visual_owner = self.text_encoder.model
            elif hasattr(self.text_encoder, "visual"):
                visual_owner = self.text_encoder
            if visual_owner is not None:
                del visual_owner.visual
            else:
                logger.warning("Qwen-Image: vision tower not found on text encoder; skipping drop")
            self.text_encoder = self.text_encoder.to(self.device)
        self.vae = (
            from_pretrained_with_prefetch(
                DistributedAutoencoderKLQwenImage.from_pretrained,
                model,
                subfolder="vae",
                prefetch_list=qwen_subfolders,
                local_files_only=local_files_only,
            ).to(self.device)
            if "vae" in owned_components
            else None
        )

        if owns_transformer:
            transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, QwenImageTransformer2DModel)
            self.transformer = QwenImageTransformer2DModel(
                od_config=od_config, quant_config=od_config.quantization_config, **transformer_kwargs
            )
            self.transformer_in_channels = self.transformer.in_channels
            self.transformer_guidance_embeds = self.transformer.guidance_embeds
        else:
            self.transformer = None
            self.transformer_in_channels = int(self._get_tf_config_value("in_channels", 64))
            self.transformer_guidance_embeds = bool(self._get_tf_config_value("guidance_embeds", False))

        self.tokenizer = (
            Qwen2Tokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
            if "tokenizer" in owned_components
            else None
        )

        self.stage = self.stage_role

        self.vae_scale_factor = (
            2 ** len(self.vae.temperal_downsample) if self.vae is not None else self.DEFAULT_VAE_SCALE_FACTOR
        )
        # QwenImage latents are turned into 2x2 patches and packed.
        # This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        # self.image_processor = VaeImageProcessor(
        #     vae_scale_factor=self.vae_scale_factor * 2
        # )
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"  # noqa: E501
        self.prompt_template_encode_start_idx = 34
        self.default_sample_size = 128
        self._guidance_scale = 0.0
        self._attention_kwargs = {}
        self._num_timesteps = 0
        self._current_timestep = None
        self._interrupt = False

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    @classmethod
    def _resolve_stage_role(cls, od_config: OmniDiffusionConfig) -> str:
        # Validate the role before component discovery and loading use it.
        role = getattr(od_config, "stage_role", "all")
        if role not in cls._ROLE_COMPONENTS:
            raise ValueError(f"Unsupported QwenImage stage_role: {role}")
        return role

    def _get_tf_config_value(self, key: str, default: Any) -> Any:
        # Read transformer config when the transformer module is not loaded in this role.
        tf_config = getattr(self.od_config, "tf_model_config", None)
        getter = getattr(tf_config, "get", None)
        return getter(key, default) if callable(getter) else default

    def _check_inputs_impl(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} "
                f"but are {height} and {width}. Dimensions will be resized accordingly"
            )

        # if callback_on_step_end_tensor_inputs is not None and not all(
        #     k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        # ):
        #     raise ValueError(
        #         f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs},
        # but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
        #     )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `prompt_embeds_mask` also have to be passed. "
                "Make sure to generate `prompt_embeds_mask` from the same text encoder "
                "that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` also have to be passed. "
                "Make sure to generate `negative_prompt_embeds_mask` from the same text encoder "
                "that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > self.tokenizer_max_length:
            raise ValueError(
                f"`max_sequence_length` cannot be greater than {self.tokenizer_max_length} but is {max_sequence_length}"
            )

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        dtype: torch.dtype | None = None,
        max_sequence_length: int | None = None,
        prompt_name: str = "prompt",
    ):
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt,
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(self.device)
        # Validate only the user prompt contribution. The Qwen template also
        # adds a fixed suffix after the user text, so subtracting only
        # prompt_template_encode_start_idx would overcount near-limit prompts.
        template_tokens = self.tokenizer(
            [template.format("")],
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(self.device)
        validate_prompt_sequence_lengths(
            txt_tokens.attention_mask,
            max_sequence_length=max_sequence_length or self.tokenizer_max_length,
            supported_max_sequence_length=self.tokenizer_max_length,
            prompt_name=prompt_name,
            baseline_attention_mask=template_tokens.attention_mask,
            error_context="after applying the Qwen prompt template",
        )
        encoder_hidden_states = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype)

        return prompt_embeds, encoder_attention_mask

    def _encode_prompt_impl(
        self,
        prompt: str | list[str],
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
        prompt_name: str = "prompt",
    ):
        r"""

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """

        # Keep the original prompt encoder as an atom helper for encode_stage.
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            if self.text_encoder is None or self.tokenizer is None:
                raise RuntimeError("QwenImage encode_stage requires text_encoder and tokenizer loaded.")
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt,
                max_sequence_length=max_sequence_length,
                prompt_name=prompt_name,
            )

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

        return latents

    def _prepare_latents_impl(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ) -> torch.Tensor:
        # Keep the original latent initializer as an atom helper for prepare_latents.
        # generator=torch.Generator(device="cuda").manual_seed(42)
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        return latents

    def _prepare_timesteps_impl(self, num_inference_steps, sigmas, image_seq_len):
        # Keep timestep materialization reusable by both request and staged execution.
        if self.scheduler is None:
            raise RuntimeError("QwenImage timestep preparation requires scheduler loaded.")
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        # image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            mu=mu,
        )
        return timesteps, num_inference_steps

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def _extract_prompts(self, prompts):
        """Extract prompt and negative_prompt from OmniPromptType list."""
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in prompts] or None
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in prompts):
            negative_prompt = None
        elif prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in prompts]
        else:
            negative_prompt = None
        return prompt, negative_prompt

    def _stage_inputs_from_state(
        self,
        state: "DiffusionRequestState",
    ) -> dict[str, Any]:
        # Normalize request inputs once so encode atoms consume a shared view.
        if self.text_encoder is None or self.tokenizer is None or self.scheduler is None:
            raise RuntimeError("QwenImage encode_stage requires scheduler, text_encoder, and tokenizer loaded.")
        # Cache parsed inputs so all encode atoms see the same normalized values.
        sampling = state.sampling
        prompt, extracted_negative_prompt = self._extract_prompts(state.prompts or [])
        negative_prompt = extracted_negative_prompt
        height = sampling.height or self.default_sample_size * self.vae_scale_factor
        width = sampling.width or self.default_sample_size * self.vae_scale_factor
        height, width = normalize_min_aligned_size(height, width, self.vae_scale_factor * 2)
        guidance_scale = sampling.guidance_scale if sampling.guidance_scale_provided else 1.0
        num_images_per_prompt = sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else 1
        inputs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": sampling.num_inference_steps or 50,
            "sigmas": sampling.sigmas,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images_per_prompt,
            "generator": sampling.generator,
            "true_cfg_scale": sampling.true_cfg_scale or 4.0,
            "max_sequence_length": sampling.max_sequence_length or self.tokenizer_max_length,
            "prompt_embeds": None,
            "prompt_embeds_mask": None,
            "negative_prompt_embeds": None,
            "negative_prompt_embeds_mask": None,
            "latents": getattr(sampling, "latents", None),
            "attention_kwargs": None,
            "callback_on_step_end_tensor_inputs": None,
        }
        state.extra["_qwen_image_stage_inputs"] = inputs
        return inputs

    def check_inputs(self, state: "DiffusionRequestState") -> None:
        """Validate request inputs and initialize request-local encode context."""
        # Atom 1: validate request fields and store encode-local scalar settings.
        inputs = self._stage_inputs_from_state(state)
        self._check_inputs_impl(
            inputs["prompt"],
            inputs["height"],
            inputs["width"],
            inputs["negative_prompt"],
            inputs["prompt_embeds"],
            inputs["negative_prompt_embeds"],
            inputs["prompt_embeds_mask"],
            inputs["negative_prompt_embeds_mask"],
            inputs["callback_on_step_end_tensor_inputs"],
            inputs["max_sequence_length"],
        )
        self._guidance_scale = inputs["guidance_scale"]
        self._attention_kwargs = inputs["attention_kwargs"] or {}
        self._current_timestep = None
        self._interrupt = False

    def encode_prompt(self, state: "DiffusionRequestState") -> None:
        """Encode positive and optional negative prompts into the request state."""
        # Atom 2: write prompt embeddings and CFG metadata into the shared state.
        inputs = state.extra["_qwen_image_stage_inputs"]
        prompt = inputs["prompt"]
        prompt_embeds = inputs["prompt_embeds"]
        negative_prompt = inputs["negative_prompt"]
        negative_prompt_embeds = inputs["negative_prompt_embeds"]
        negative_prompt_embeds_mask = inputs["negative_prompt_embeds_mask"]

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            batch_size = 1

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = inputs["true_cfg_scale"] > 1 and has_neg_prompt
        self.check_cfg_parallel_validity(inputs["true_cfg_scale"], has_neg_prompt)

        prompt_embeds, prompt_embeds_mask = self._encode_prompt_impl(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=inputs["prompt_embeds_mask"],
            num_images_per_prompt=inputs["num_images_per_prompt"],
            max_sequence_length=inputs["max_sequence_length"],
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self._encode_prompt_impl(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                num_images_per_prompt=inputs["num_images_per_prompt"],
                max_sequence_length=inputs["max_sequence_length"],
                prompt_name="negative_prompt",
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

        state.prompt_embeds = prompt_embeds
        state.prompt_embeds_mask = prompt_embeds_mask
        state.negative_prompt_embeds = negative_prompt_embeds
        state.negative_prompt_embeds_mask = negative_prompt_embeds_mask
        state.do_true_cfg = do_true_cfg
        state.txt_seq_lens = txt_seq_lens_from_embeds(prompt_embeds)
        state.negative_txt_seq_lens = txt_seq_lens_from_embeds(negative_prompt_embeds)
        state.extra["_qwen_image_batch_size"] = batch_size

    def prepare_latents(self, state: "DiffusionRequestState") -> None:
        """Initialize packed latents and image metadata for the request."""
        # Atom 3: create packed latents and image shape metadata needed by DiT.
        inputs = state.extra["_qwen_image_stage_inputs"]
        batch_size = int(state.extra["_qwen_image_batch_size"])
        num_channels_latents = self.transformer_in_channels // 4
        latents = self._prepare_latents_impl(
            batch_size * inputs["num_images_per_prompt"],
            num_channels_latents,
            inputs["height"],
            inputs["width"],
            state.prompt_embeds.dtype,
            self.device,
            inputs["generator"],
            inputs["latents"],
        )

        state.latents = latents
        state.img_shapes = [
            [(1, inputs["height"] // self.vae_scale_factor // 2, inputs["width"] // self.vae_scale_factor // 2)]
        ] * batch_size
        if self.transformer_guidance_embeds:
            guidance = torch.full([1], inputs["guidance_scale"], dtype=torch.float32)
            state.guidance = guidance.expand(latents.shape[0])
        else:
            state.guidance = None

    def prepare_timesteps(self, state: "DiffusionRequestState") -> None:
        """Materialize timesteps and a per-request scheduler."""
        # Atom 4: attach request-local scheduler state for later DiT steps.
        inputs = state.extra["_qwen_image_stage_inputs"]
        timesteps, _ = self._prepare_timesteps_impl(
            inputs["num_inference_steps"],
            inputs["sigmas"],
            state.latents.shape[1],
        )
        self._num_timesteps = len(timesteps)
        req_scheduler = copy.deepcopy(self.scheduler)
        req_scheduler.set_begin_index(0)
        state.step_index = 0
        state.scheduler = req_scheduler
        state.timesteps = timesteps
        # QwenImage always normalizes CFG output (matching forward())
        state.sampling.cfg_normalize = True
        state.extra.pop("_qwen_image_stage_inputs", None)
        state.extra.pop("_qwen_image_batch_size", None)

    def _build_denoise_kwargs(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        img_shapes: list,
        txt_seq_lens: list[int] | None,
        do_true_cfg: bool,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds_mask: torch.Tensor | None,
        negative_txt_seq_lens: list[int] | None,
        image_latents: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, int | None]:
        """Build positive/negative kwargs and output_slice for one denoise step.

        Returns:
            (positive_kwargs, negative_kwargs, output_slice)
        """
        transformer_kwargs = {
            "attention_kwargs": self.attention_kwargs,
            "return_dict": False,
        }

        # Broadcast timestep to match batch size
        t_for_model = timestep.expand(latents.shape[0]).to(
            device=latents.device,
            dtype=latents.dtype,
        )

        # Concatenate image latents if available (editing pipelines)
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
            **transformer_kwargs,
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
                **transformer_kwargs,
            }
        else:
            negative_kwargs = None

        output_slice = latents.size(1) if image_latents is not None else None
        return positive_kwargs, negative_kwargs, output_slice

    def _decode_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        output_type: str = "pil",
    ) -> DiffusionOutput:
        """Unpack, normalize, and VAE-decode latents into a DiffusionOutput."""
        if output_type == "latent":
            return DiffusionOutput(
                output=latents,
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )

        if self.vae is None:
            raise RuntimeError("QwenImage decode_stage requires VAE loaded.")
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

    def _predict_noise_impl(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        # Isolate the raw transformer call so CFGParallelMixin can reuse it.
        if self.transformer is None:
            raise RuntimeError("QwenImage denoise_stage requires transformer loaded.")
        result = self.transformer(*args, **kwargs)
        return result if isinstance(result, IntermediateTensors) else result[0]

    def predict_noise(
        self,
        input_batch: "InputBatch",
    ) -> torch.Tensor | None:
        """One denoise step: read from *input_batch*, delegate to CFGParallelMixin.

        Reuses ``predict_noise_maybe_with_cfg`` so that CFG-parallel,
        sequential-CFG, and no-CFG paths are handled identically to
        ``diffuse()``.
        """
        # DiT atom: build batched transformer inputs and delegate CFG handling.
        if self.interrupt:
            return None
        if self.transformer is None:
            raise RuntimeError("QwenImage denoise_stage requires transformer loaded.")

        t = input_batch.timesteps
        self._current_timestep = t
        self.transformer.do_true_cfg = input_batch.do_true_cfg

        positive_kwargs, negative_kwargs, output_slice = self._build_denoise_kwargs(
            latents=input_batch.latents,
            timestep=t,
            guidance=input_batch.guidance,
            prompt_embeds=input_batch.prompt_embeds,
            prompt_embeds_mask=input_batch.prompt_embeds_mask,
            img_shapes=input_batch.img_shapes,
            txt_seq_lens=input_batch.txt_seq_lens,
            do_true_cfg=input_batch.do_true_cfg,
            negative_prompt_embeds=input_batch.negative_prompt_embeds,
            negative_prompt_embeds_mask=input_batch.negative_prompt_embeds_mask,
            negative_txt_seq_lens=input_batch.negative_txt_seq_lens,
            image_latents=input_batch.image_latents,
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
        state: "DiffusionRequestState",
        noise_pred: torch.Tensor,
    ) -> None:
        """One scheduler step: update ``state.latents`` and advance ``step_index``."""
        # Scheduler atom: mutate only the current request state.
        if self.interrupt:
            return

        t = state.current_timestep
        state.latents = self.scheduler_step_maybe_with_cfg(
            noise_pred,
            t,
            state.latents,
            state.do_true_cfg,
            per_request_scheduler=state.scheduler,
        )

        state.step_index += 1

    def decode(
        self,
        state: "DiffusionRequestState",
    ) -> DiffusionOutput:
        """Decode final latents from *state*."""
        # Decode atom: derive output shape from the request state.
        self._current_timestep = None

        height = state.sampling.height or self.default_sample_size * self.vae_scale_factor
        width = state.sampling.width or self.default_sample_size * self.vae_scale_factor
        height, width = normalize_min_aligned_size(height, width, self.vae_scale_factor * 2)
        output_type = getattr(state.sampling, "output_type", None) or "pil"

        return self._decode_latents(state.latents, height, width, output_type)

    def rehydrate_stage_state(self, state: "DiffusionRequestState") -> "DiffusionRequestState":
        """Rebuild process-local scheduler state after transport."""
        # Transport hook: recreate non-serializable scheduler objects in the receiving role.
        if self.scheduler is None:
            return state
        if state.scheduler is not None or state.timesteps is None or state.latents is None:
            return state
        timesteps = state.timesteps
        self._prepare_timesteps_impl(
            state.total_steps,
            state.sampling.sigmas,
            state.latents.shape[1],
        )
        req_scheduler = copy.deepcopy(self.scheduler)
        req_scheduler.set_begin_index(state.step_index)
        state.scheduler = req_scheduler
        state.timesteps = timesteps
        return state

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Encoder/decode-only roles load all owned weights during component initialization.
        if self.weights_loaded_by_model_init:
            return {name for name, _ in self.named_parameters()}
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


class QwenImageDMD2Pipeline(DMD2PipelineMixin, QwenImagePipeline):
    """QwenImage pipeline for FastGen DMD2-distilled models."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()
