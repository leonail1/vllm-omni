# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import inspect
import json
import logging
import math
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import PIL.Image
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.composed_pipeline import ComposedDiffusionPipeline
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.qwen_image.cfg_parallel import (
    QwenImageCFGParallelMixin,
)
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import calculate_shift
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageTransformer2DModel,
)
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.prompt_utils import (
    validate_prompt_sequence_lengths,
)
from vllm_omni.diffusion.utils.size_utils import (
    normalize_min_aligned_size,
)
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import DiffusionRequestState

logger = logging.getLogger(__name__)


def get_qwen_image_edit_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-processing function for QwenImageEditPipeline."""
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])
    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** len(vae_config["temporal_downsample"]) if "temporal_downsample" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2, do_convert_rgb=True)
    latent_channels = vae_config.get("z_dim", 16)

    def pre_process_func(
        request: OmniDiffusionRequest,
    ):
        """Pre-process requests for QwenImageEditPipeline."""
        for i, prompt in enumerate(request.prompts):
            multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
            raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
            if isinstance(prompt, str):
                prompt = OmniTextPrompt(prompt=prompt)
            if "additional_information" not in prompt:
                prompt["additional_information"] = {}

            # Only handles single image
            if not raw_image:  # None or empty list
                raise ValueError("""Received no input image. This model requires one input image to run.""")
            elif isinstance(raw_image, list):
                if len(raw_image) > 1:
                    raise ValueError(
                        """Received multiple input images. Only a single image is supported by this model."""
                    )
                else:
                    raw_image = raw_image[0]

            if isinstance(raw_image, str):
                image = PIL.Image.open(raw_image)
            else:
                image = cast(PIL.Image.Image | torch.Tensor | np.ndarray, raw_image)

            image_size = image.size
            calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
            height = request.sampling_params.height or calculated_height
            width = request.sampling_params.width or calculated_width

            # Ensure dimensions are multiples of vae_scale_factor * 2
            height, width = normalize_min_aligned_size(height, width, vae_scale_factor * 2)

            # Store calculated dimensions in request
            prompt["additional_information"]["calculated_height"] = calculated_height
            prompt["additional_information"]["calculated_width"] = calculated_width
            request.sampling_params.height = height
            request.sampling_params.width = width

            # Preprocess image
            if image is not None and not (
                isinstance(image, torch.Tensor) and len(image.shape) > 1 and image.shape[1] == latent_channels
            ):
                image = image_processor.resize(image, calculated_height, calculated_width)
                prompt_image = image
                image = image_processor.preprocess(image, calculated_height, calculated_width)
                image = image.unsqueeze(2)

                # Store preprocessed image and prompt image in request
                prompt["additional_information"]["preprocessed_image"] = image
                prompt["additional_information"]["prompt_image"] = prompt_image
            request.prompts[i] = prompt
        return request

    return pre_process_func


def get_qwen_image_edit_post_process_func(
    od_config: OmniDiffusionConfig,
):
    """Post-processing function for QwenImageEditPipeline."""
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])
    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** len(vae_config["temporal_downsample"]) if "temporal_downsample" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2, do_convert_rgb=True)

    def post_process_func(
        images: torch.Tensor,
    ):
        return image_processor.postprocess(images)

    return post_process_func


def calculate_dimensions(target_area: float, ratio: float):
    """Calculate width and height from target area and aspect ratio."""
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height


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


def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "argmax"
):
    """Retrieve latents from VAE encoder output."""
    if hasattr(encoder_output, "latent_dist"):
        return (
            encoder_output.latent_dist.mode()
            if sample_mode == "argmax"
            else encoder_output.latent_dist.sample(generator)
        )
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class QwenImageEditPipeline(
    ComposedDiffusionPipeline,
    nn.Module,
    SupportImageInput,
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
        "prepare_latents",
        "prepare_timesteps",
    )
    DEFAULT_VAE_SCALE_FACTOR: ClassVar[int] = 8
    DEFAULT_LATENT_CHANNELS: ClassVar[int] = 16
    _ROLE_COMPONENTS: ClassVar[dict[str, set[str]]] = {
        "all": {"scheduler", "text_encoder", "tokenizer", "processor", "vae", "transformer"},
        "encode": {"scheduler", "text_encoder", "tokenizer", "processor", "vae"},
        "dit": {"scheduler", "transformer"},
        "decode": {"vae"},
    }

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        model = od_config.model
        self.stage_role = self._resolve_stage_role(od_config)
        owned_components = self._ROLE_COMPONENTS[self.stage_role]
        owns_transformer = "transformer" in owned_components

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
            subfolder
            for subfolder in ("scheduler", "text_encoder", "vae", "tokenizer", "processor")
            if subfolder in owned_components
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
            ).to(self.device)
            if "text_encoder" in owned_components
            else None
        )

        self.vae = (
            from_pretrained_with_prefetch(
                AutoencoderKLQwenImage.from_pretrained,
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
            self.transformer = QwenImageTransformer2DModel(od_config=od_config, **transformer_kwargs)
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
        self.processor = (
            from_pretrained_with_prefetch(
                Qwen2VLProcessor.from_pretrained,
                model,
                subfolder="processor",
                prefetch_list=qwen_subfolders,
                local_files_only=local_files_only,
            )
            if "processor" in owned_components
            else None
        )

        self.stage = self.stage_role

        self.vae_scale_factor = (
            2 ** len(self.vae.temperal_downsample) if self.vae is not None else self.DEFAULT_VAE_SCALE_FACTOR
        )
        self.latent_channels = self.vae.config.z_dim if self.vae is not None else self.DEFAULT_LATENT_CHANNELS
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2, do_convert_rgb=True)
        self.tokenizer_max_length = 1024
        # Edit prompt template - different from generation template
        self.prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"  # noqa: E501
        self.prompt_template_encode_start_idx = 64
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
        role = getattr(od_config, "stage_role", "all")
        if role not in cls._ROLE_COMPONENTS:
            raise ValueError(f"Unsupported QwenImageEdit stage_role: {role}")
        return role

    def _get_tf_config_value(self, key: str, default: Any) -> Any:
        tf_config = getattr(self.od_config, "tf_model_config", None)
        getter = getattr(tf_config, "get", None)
        return getter(key, default) if callable(getter) else default

    def _check_inputs_impl(
        self,
        prompt,
        height,
        width,
        image=None,
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
                f" {negative_prompt_embeds}. Make sure to only forward one of the two."
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
        image: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
    ):
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]

        model_inputs = self.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=self.device)

        return prompt_embeds, encoder_attention_mask

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        image: PIL.Image.Image | torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        max_sequence_length: int | None = None,
        prompt_name: str = "prompt",
    ):
        """Get prompt embeddings with image support for editing."""
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
        # The edit template contains fixed multimodal scaffolding around the
        # instruction. Validate against the empty-template baseline so image
        # placeholder text does not consume the user's text budget.
        template_tokens = self.tokenizer(
            [template.format("")],
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(self.device)
        # Qwen-Image-Edit expands image placeholders into many vision tokens
        # inside the processor. `max_sequence_length` is meant to constrain the
        # prompt text length, so validate on the text template before image
        # token expansion.
        validate_prompt_sequence_lengths(
            txt_tokens.attention_mask,
            max_sequence_length=max_sequence_length or self.tokenizer_max_length,
            supported_max_sequence_length=self.tokenizer_max_length,
            prompt_name=prompt_name,
            baseline_attention_mask=template_tokens.attention_mask,
            error_context="after applying the Qwen prompt template",
        )

        # Use processor to handle both text and image inputs
        model_inputs = self.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
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
        image: torch.Tensor | None = None,
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
            image (`torch.Tensor`, *optional*):
                image to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt,
                image,
                max_sequence_length=max_sequence_length,
                prompt_name=prompt_name,
            )

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

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i], sample_mode="argmax")
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std

        return image_latents

    def _prepare_latents_impl(
        self,
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height, width)

        image_latents = None
        if image is not None:
            image = image.to(device=device, dtype=dtype)
            if image.shape[1] != self.latent_channels:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            else:
                image_latents = image
            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                image_latents = torch.cat([image_latents], dim=0)

            image_latent_height, image_latent_width = image_latents.shape[3:]
            image_latents = self._pack_latents(
                image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
            )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents

    def _prepare_timesteps_impl(self, num_inference_steps, sigmas, image_seq_len):
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
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

    def _stage_inputs_from_state(
        self,
        state: "DiffusionRequestState",
    ) -> dict[str, Any]:
        if self.text_encoder is None or self.tokenizer is None or self.processor is None or self.scheduler is None:
            raise RuntimeError("QwenImageEdit encode_stage requires scheduler, text_encoder, tokenizer, and processor.")

        sampling = state.sampling
        # TODO: In online mode, sometimes it receives [{"negative_prompt": None}, {...}], so cannot use .get("...", "")
        # TODO: May be some data formatting operations on the API side. Hack for now.
        if len(state.prompts) > 1:
            logger.warning(
                """This model only supports a single prompt, not a batched request.""",
                """Taking only the first image for now.""",
            )
        first_prompt = state.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
        negative_prompt = None if isinstance(first_prompt, str) else first_prompt.get("negative_prompt")
        if negative_prompt is None:
            logger.warning(
                "negative_prompt is not set. The official Qwen-Image-Edit model "
                "may produce lower-quality results without a negative_prompt. "
                "Qwen official repository recommends to use whitespace string as negative_prompt. "
                "Note: some distilled variants may not be affected by this."
            )

        # Get preprocessed image from request (pre-processing is done in DiffusionEngine)
        if not isinstance(first_prompt, str) and "preprocessed_image" in (
            additional_information := first_prompt.get("additional_information", {})
        ):
            prompt_image = additional_information.get("prompt_image")
            image = additional_information.get("preprocessed_image")
            calculated_height = additional_information.get("calculated_height")
            calculated_width = additional_information.get("calculated_width")
            height = sampling.height
            width = sampling.width
        else:
            # fallback to run pre-processing in pipeline (debug only)
            image = None
            image_size = image[0].size if isinstance(image, list) else image.size
            calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
            height = sampling.height or calculated_height
            width = sampling.width or calculated_width

            height, width = normalize_min_aligned_size(height, width, self.vae_scale_factor * 2)

            if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
                image = self.image_processor.resize(image, calculated_height, calculated_width)
                prompt_image = image
                image = self.image_processor.preprocess(image, calculated_height, calculated_width)
                image = image.unsqueeze(2)

        num_inference_steps = sampling.num_inference_steps or 50
        sigmas = sampling.sigmas
        max_sequence_length = sampling.max_sequence_length or self.tokenizer_max_length
        generator = sampling.generator
        true_cfg_scale = sampling.true_cfg_scale or 4.0
        guidance_scale = 1.0
        if sampling.guidance_scale_provided:
            guidance_scale = sampling.guidance_scale
        num_images_per_prompt = (
            sampling.num_outputs_per_prompt
            if sampling.num_outputs_per_prompt > 0
            else 1
        )

        sampling.height = height
        sampling.width = width
        inputs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "prompt_image": prompt_image,
            "image": image,
            "calculated_height": calculated_height,
            "calculated_width": calculated_width,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "sigmas": sigmas,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images_per_prompt,
            "generator": generator,
            "true_cfg_scale": true_cfg_scale,
            "max_sequence_length": max_sequence_length,
            "prompt_embeds": None,
            "prompt_embeds_mask": None,
            "negative_prompt_embeds": None,
            "negative_prompt_embeds_mask": None,
            "latents": getattr(sampling, "latents", None),
            "output_type": sampling.output_type or "pil",
            "attention_kwargs": None,
            "callback_on_step_end_tensor_inputs": None,
        }
        sampling.output_type = inputs["output_type"]
        state.extra["_qwen_image_edit_stage_inputs"] = inputs
        return inputs

    def check_inputs(self, state: "DiffusionRequestState") -> None:
        # Atom 1: normalize request fields and run the original edit input checks.
        inputs = self._stage_inputs_from_state(state)
        self._check_inputs_impl(
            inputs["prompt"],
            inputs["height"],
            inputs["width"],
            inputs["image"],
            inputs["negative_prompt"],
            inputs["prompt_embeds"],
            inputs["negative_prompt_embeds"],
            inputs["prompt_embeds_mask"],
            inputs["negative_prompt_embeds_mask"],
            inputs["callback_on_step_end_tensor_inputs"],
            inputs["max_sequence_length"],
        )

        self._guidance_scale = inputs["guidance_scale"]
        self._attention_kwargs = inputs["attention_kwargs"]
        self._current_timestep = None
        self._interrupt = False

    def encode_prompt(self, state: "DiffusionRequestState") -> None:
        # Atom 2: encode positive/negative image-conditioned prompts.
        inputs = state.extra["_qwen_image_edit_stage_inputs"]
        prompt = inputs["prompt"]
        prompt_embeds = inputs["prompt_embeds"]
        negative_prompt = inputs["negative_prompt"]
        negative_prompt_embeds = inputs["negative_prompt_embeds"]
        negative_prompt_embeds_mask = inputs["negative_prompt_embeds_mask"]

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )

        true_cfg_scale = inputs["true_cfg_scale"]
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        self.check_cfg_parallel_validity(true_cfg_scale, has_neg_prompt)

        prompt_embeds, prompt_embeds_mask = self._encode_prompt_impl(
            prompt=prompt,
            image=inputs["prompt_image"],  # Use resized image for prompt encoding
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=inputs["prompt_embeds_mask"],
            num_images_per_prompt=inputs["num_images_per_prompt"],
            max_sequence_length=inputs["max_sequence_length"],
        )

        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self._encode_prompt_impl(
                prompt=negative_prompt,
                image=inputs["prompt_image"],  # Use same resized image for negative prompt encoding
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
        state.txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        state.negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )
        state.extra["_qwen_image_edit_batch_size"] = batch_size

    def prepare_latents(self, state: "DiffusionRequestState") -> None:
        # Atom 3: create random latents and encode/pack the conditioning image latents.
        inputs = state.extra["_qwen_image_edit_stage_inputs"]
        batch_size = int(state.extra["_qwen_image_edit_batch_size"])

        num_channels_latents = self.transformer_in_channels // 4
        # random noise latents, and image latents encoded by vae
        latents, image_latents = self._prepare_latents_impl(
            inputs["image"],
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
        state.sampling.image_latent = image_latents
        state.img_shapes = [
            [
                (1, inputs["height"] // self.vae_scale_factor // 2, inputs["width"] // self.vae_scale_factor // 2),
                (
                    1,
                    inputs["calculated_height"] // self.vae_scale_factor // 2,
                    inputs["calculated_width"] // self.vae_scale_factor // 2,
                ),
            ]
        ] * batch_size

        # handle guidance
        if self.transformer_guidance_embeds:
            guidance = torch.full([1], inputs["guidance_scale"], dtype=torch.float32)
            state.guidance = guidance.expand(latents.shape[0])
        else:
            state.guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

    def prepare_timesteps(self, state: "DiffusionRequestState") -> None:
        # Atom 4: materialize timesteps and request-local scheduler state.
        inputs = state.extra["_qwen_image_edit_stage_inputs"]
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
        state.sampling.cfg_normalize = True
        state.extra.pop("_qwen_image_edit_stage_inputs", None)
        state.extra.pop("_qwen_image_edit_batch_size", None)

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
        additional_transformer_kwargs = {
            "return_dict": False,
            "attention_kwargs": self.attention_kwargs,
        }

        timestep = timestep.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)

        latent_model_input = latents
        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1)

        positive_kwargs = {
            "hidden_states": latent_model_input,
            "timestep": timestep / 1000,
            "guidance": guidance,
            "encoder_hidden_states_mask": prompt_embeds_mask,
            "encoder_hidden_states": prompt_embeds,
            "img_shapes": img_shapes,
            "txt_seq_lens": txt_seq_lens,
            **additional_transformer_kwargs,
        }
        if do_true_cfg:
            negative_kwargs = {
                "hidden_states": latent_model_input,
                "timestep": timestep / 1000,
                "guidance": guidance,
                "encoder_hidden_states_mask": negative_prompt_embeds_mask,
                "encoder_hidden_states": negative_prompt_embeds,
                "img_shapes": img_shapes,
                "txt_seq_lens": negative_txt_seq_lens,
                **additional_transformer_kwargs,
            }
        else:
            negative_kwargs = None

        # For editing pipelines, slice the output to remove condition latents.
        output_slice = latents.size(1) if image_latents is not None else None
        return positive_kwargs, negative_kwargs, output_slice

    def _decode_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        output_type: str = "pil",
    ) -> DiffusionOutput:
        if output_type == "latent":
            return DiffusionOutput(
                output=latents,
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )

        if self.vae is None:
            raise RuntimeError("QwenImageEdit decode_stage requires VAE loaded.")
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
            output=image, stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None
        )

    def _predict_noise_impl(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        if self.transformer is None:
            raise RuntimeError("QwenImageEdit denoise_stage requires transformer loaded.")
        result = self.transformer(*args, **kwargs)
        return result if isinstance(result, IntermediateTensors) else result[0]

    def predict_noise(
        self,
        input_batch: "InputBatch",
    ) -> torch.Tensor | None:
        # DiT atom: run one original diffuse-loop transformer step.
        if self.interrupt:
            return None
        if self.transformer is None:
            raise RuntimeError("QwenImageEdit denoise_stage requires transformer loaded.")

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
        # Scheduler atom: run the original diffuse-loop scheduler step.
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
        self._current_timestep = None

        height = state.sampling.height or self.default_sample_size * self.vae_scale_factor
        width = state.sampling.width or self.default_sample_size * self.vae_scale_factor
        output_type = state.sampling.output_type or "pil"
        return self._decode_latents(state.latents, height, width, output_type)

    def rehydrate_stage_state(self, state: "DiffusionRequestState") -> "DiffusionRequestState":
        # Transport hook: recreate the scheduler object in the receiving role.
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
        if self.weights_loaded_by_model_init:
            return {name for name, _ in self.named_parameters()}
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
