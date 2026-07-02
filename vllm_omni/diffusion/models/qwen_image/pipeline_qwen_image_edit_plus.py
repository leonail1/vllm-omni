# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
import logging
import os
from collections.abc import Iterable
from typing import Any, cast

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

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.model_metadata import QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.models.qwen_image.cfg_parallel import (
    QwenImageCFGParallelMixin,
)
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import calculate_shift
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit import (
    calculate_dimensions,
    retrieve_latents,
    retrieve_timesteps,
)
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageTransformer2DModel,
)
from vllm_omni.diffusion.models.qwen_image.step_batch import (
    QwenImageStepAtomsMixin,
    get_qwen_state,
    load_qwen_transformer_weights,
    qwen_diffusion_stage_role,
    qwen_role_loads_condition_vae,
    qwen_role_loads_decoder_vae,
    qwen_role_loads_scheduler,
    qwen_role_loads_text_encoder,
    qwen_role_loads_transformer,
    qwen_transformer_guidance_embeds,
    qwen_transformer_in_channels,
    qwen_vae_metadata,
    set_qwen_state,
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
from vllm_omni.diffusion.worker.utils import DiffusionRequestState
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

logger = logging.getLogger(__name__)

CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024
# Keep this in sync with the practical conditioning-token budget for
# Qwen-Image-Edit-2511. Empirically, 4 images stays within the supported range
# while 5 images overflows the prompt/conditioning path and fails downstream.
# Re-export the shared metadata value locally so this pipeline keeps a nearby,
# descriptive constant for validation and tests without becoming the source of truth.
MAX_QWEN_IMAGE_EDIT_PLUS_INPUT_IMAGES = QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES


def get_qwen_image_edit_plus_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-processing function for QwenImageEditPlusPipeline."""
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
        """Pre-process requests for QwenImageEditPlusPipeline."""
        for i, prompt in enumerate(request.prompts):
            multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
            raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
            if isinstance(prompt, str):
                prompt = OmniTextPrompt(prompt=prompt)
            if "additional_information" not in prompt:
                prompt["additional_information"] = {}

            # Handle single image or list of images
            if raw_image is None:
                continue

            if not isinstance(raw_image, list):
                raw_image = [raw_image]
            if len(raw_image) > MAX_QWEN_IMAGE_EDIT_PLUS_INPUT_IMAGES:
                raise ValueError(
                    f"Received {len(raw_image)} input images. "
                    f"At most {MAX_QWEN_IMAGE_EDIT_PLUS_INPUT_IMAGES} images are supported by this model."
                )
            image = [
                PIL.Image.open(im) if isinstance(im, str) else cast(PIL.Image.Image | np.ndarray | torch.Tensor, im)
                for im in raw_image
            ]

            # Calculate dimensions based on first image
            image_size = image[0].size
            calculated_width, calculated_height = calculate_dimensions(VAE_IMAGE_SIZE, image_size[0] / image_size[1])
            height = request.sampling_params.height or calculated_height
            width = request.sampling_params.width or calculated_width

            # Ensure dimensions are multiples of vae_scale_factor * 2
            height, width = normalize_min_aligned_size(height, width, vae_scale_factor * 2)

            # Store calculated dimensions in request
            prompt["additional_information"]["calculated_height"] = calculated_height
            prompt["additional_information"]["calculated_width"] = calculated_width
            request.sampling_params.height = height
            request.sampling_params.width = width

            # Preprocess images into condition_images (for prompt encoding) and vae_images (for VAE encoding)
            condition_images = []
            vae_images = []
            condition_image_sizes = []
            vae_image_sizes = []

            for img in image:
                if isinstance(img, torch.Tensor) and len(img.shape) > 1 and img.shape[1] == latent_channels:
                    # Already a latent tensor
                    continue

                image_width, image_height = img.size
                condition_width, condition_height = calculate_dimensions(
                    CONDITION_IMAGE_SIZE, image_width / image_height
                )
                vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)

                condition_image_sizes.append((condition_width, condition_height))
                vae_image_sizes.append((vae_width, vae_height))

                condition_images.append(image_processor.resize(img, condition_height, condition_width))
                vae_images.append(image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))

            # Store preprocessed images in request
            prompt["additional_information"]["condition_images"] = condition_images
            prompt["additional_information"]["vae_images"] = vae_images
            prompt["additional_information"]["condition_image_sizes"] = condition_image_sizes
            prompt["additional_information"]["vae_image_sizes"] = vae_image_sizes
            request.prompts[i] = prompt
        return request

    return pre_process_func


def get_qwen_image_edit_plus_post_process_func(
    od_config: OmniDiffusionConfig,
):
    """Post-processing function for QwenImageEditPlusPipeline."""
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


class QwenImageEditPlusPipeline(
    nn.Module,
    SupportImageInput,
    QwenImageStepAtomsMixin,
    QwenImageCFGParallelMixin,
    DiffusionPipelineProfilerMixin,
):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]
        self.device = get_local_device()
        model = od_config.model
        role = qwen_diffusion_stage_role(od_config)
        self._diffusion_stage_role = role.value

        # Check if model is a local path
        local_files_only = os.path.isdir(model)
        needs_scheduler = qwen_role_loads_scheduler(role)
        needs_text_encoder = qwen_role_loads_text_encoder(role)
        needs_transformer = qwen_role_loads_transformer(role)
        needs_vae = qwen_role_loads_condition_vae(role) or qwen_role_loads_decoder_vae(role)

        # Defend against a transformers v5 multi-worker race where a peer
        # worker's partially-written shard set makes our from_pretrained
        # subfolder resolution fail with OSError (observed in Buildkite
        # #1043 for Qwen/Qwen-Image-Edit-2509). snapshot_download takes a
        # per-repo file lock so the first worker downloads and the rest
        # wait on a warm cache before loading.
        # Keep the prefetch set role-aware so split workers do not materialize
        # subfolders owned only by other roles.
        qwen_subfolders = []
        if needs_scheduler:
            qwen_subfolders.append("scheduler")
        if needs_text_encoder:
            qwen_subfolders.extend(["text_encoder", "tokenizer", "processor"])
        if needs_vae:
            qwen_subfolders.append("vae")
        prefetch_subfolders(
            model,
            qwen_subfolders,
        )

        self.scheduler = (
            FlowMatchEulerDiscreteScheduler.from_pretrained(
                model, subfolder="scheduler", local_files_only=local_files_only
            )
            if needs_scheduler
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
            if needs_text_encoder
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
            if needs_vae
            else None
        )

        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, QwenImageTransformer2DModel)
        # Store transformer metadata even when this role skips transformer
        # construction; encode/decode shape helpers still need these values.
        self._qwen_transformer_in_channels = int(transformer_kwargs.get("in_channels", 64))
        self._qwen_transformer_guidance_embeds = bool(transformer_kwargs.get("guidance_embeds", False))
        self.transformer = (
            QwenImageTransformer2DModel(od_config=od_config, **transformer_kwargs) if needs_transformer else None
        )
        if needs_text_encoder:
            self.tokenizer = Qwen2Tokenizer.from_pretrained(
                model, subfolder="tokenizer", local_files_only=local_files_only
            )
            self.processor = from_pretrained_with_prefetch(
                Qwen2VLProcessor.from_pretrained,
                model,
                subfolder="processor",
                prefetch_list=qwen_subfolders,
                local_files_only=local_files_only,
            )
        else:
            self.tokenizer = None
            self.processor = None

        self.stage = None

        fallback_scale_factor, fallback_latent_channels = qwen_vae_metadata(model)
        self.vae_scale_factor = (
            2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else fallback_scale_factor
        )
        self.latent_channels = self.vae.config.z_dim if getattr(self, "vae", None) else fallback_latent_channels
        self.image_processor = (
            VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2, do_convert_rgb=True)
            if needs_vae
            else None
        )
        self.tokenizer_max_length = 1024
        # Edit prompt template - different from generation template, supports multiple images
        self.prompt_template_encode = (
            "<|im_start|>system\nDescribe the key features of the input image "
            "(color, shape, size, texture, objects, background), then explain how the user's "
            "text instruction should alter or modify the image. Generate a new image that meets "
            "the user's requirements while maintaining consistency with the original input where "
            "appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        self.prompt_template_encode_start_idx = 64
        self.default_sample_size = 128
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def check_inputs(
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
        prompt: str | list[str],
        image: list[torch.Tensor] | torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        max_sequence_length: int | None = None,
        prompt_name: str = "prompt",
    ):
        """Get prompt embeddings with support for multiple images."""
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        # Build image prompt template for multiple images
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        if isinstance(image, list):
            base_img_prompt = ""
            for i, img in enumerate(image):
                base_img_prompt += img_prompt_template.format(i + 1)
        elif image is not None:
            base_img_prompt = img_prompt_template.format(1)
        else:
            base_img_prompt = ""

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(base_img_prompt + e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt,
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(self.device)
        # Multi-image edit prepends "Picture N" placeholders before the user
        # instruction. Subtract the placeholder-aware baseline so attached
        # images do not reduce the remaining prompt budget.
        template_tokens = self.tokenizer(
            [template.format(base_img_prompt)],
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(self.device)
        # The processor expands image placeholders into many vision tokens.
        # `max_sequence_length` should guard the prompt text length before that
        # multimodal expansion happens.
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

    def encode_prompt(
        self,
        prompt: str | list[str],
        image: list[torch.Tensor] | torch.Tensor | None = None,
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
            image (`torch.Tensor` or `list[torch.Tensor]`, *optional*):
                image(s) to be encoded
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

    def prepare_latents(
        self,
        images,
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
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            all_image_latents = []
            for image in images:
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
                all_image_latents.append(image_latents)
            # Concatenate all image latents along dimension 1
            image_latents = torch.cat(all_image_latents, dim=1)

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

    def prepare_timesteps(self, num_inference_steps, sigmas, image_seq_len):
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
        return getattr(self, "_guidance_scale", 0.0)

    @property
    def attention_kwargs(self):
        return getattr(self, "_attention_kwargs", {})

    @property
    def num_timesteps(self):
        return getattr(self, "_num_timesteps", 0)

    @property
    def current_timestep(self):
        return getattr(self, "_current_timestep", None)

    @property
    def interrupt(self):
        return getattr(self, "_interrupt", False)

    def _prepare_atom_args(self, state: DiffusionRequestState, **kwargs: Any) -> dict[str, Any]:
        if not state.prompts:
            raise ValueError("QwenImageEditPlusPipeline requires one prompt.")
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

        additional_information = first_prompt.get("additional_information", {}) if not isinstance(first_prompt, str) else {}
        required = ("vae_images", "condition_images", "vae_image_sizes", "calculated_height", "calculated_width")
        if all(key in additional_information for key in required):
            condition_images = additional_information.get("condition_images")
            vae_images = additional_information.get("vae_images")
            condition_image_sizes = additional_information.get("condition_image_sizes")
            vae_image_sizes = additional_information.get("vae_image_sizes")
            calculated_height = additional_information.get("calculated_height")
            calculated_width = additional_information.get("calculated_width")
            height = state.sampling.height or kwargs.get("height") or calculated_height
            width = state.sampling.width or kwargs.get("width") or calculated_width
        else:
            image = kwargs.get("image")
            if image is None:
                raise ValueError("QwenImageEditPlusPipeline requires images or preprocessed multi-image inputs.")
            if not isinstance(image, list):
                image = [image]
            if len(image) > MAX_QWEN_IMAGE_EDIT_PLUS_INPUT_IMAGES:
                raise ValueError(
                    f"Received {len(image)} input images. "
                    f"At most {MAX_QWEN_IMAGE_EDIT_PLUS_INPUT_IMAGES} images are supported by this model."
                )
            image_size = image[0].size
            calculated_width, calculated_height = calculate_dimensions(VAE_IMAGE_SIZE, image_size[0] / image_size[1])
            height = state.sampling.height or kwargs.get("height") or calculated_height
            width = state.sampling.width or kwargs.get("width") or calculated_width
            height, width = normalize_min_aligned_size(height, width, self.vae_scale_factor * 2)
            condition_images = []
            vae_images = []
            condition_image_sizes = []
            vae_image_sizes = []
            for img in image:
                image_width, image_height = img.size
                condition_width, condition_height = calculate_dimensions(
                    CONDITION_IMAGE_SIZE, image_width / image_height
                )
                vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
                condition_image_sizes.append((condition_width, condition_height))
                vae_image_sizes.append((vae_width, vae_height))
                condition_images.append(self.image_processor.resize(img, condition_height, condition_width))
                vae_images.append(self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))

        if height is None or width is None:
            raise ValueError("QwenImageEditPlusPipeline step execution requires output image dimensions.")

        num_images_per_prompt = state.sampling.num_outputs_per_prompt if state.sampling.num_outputs_per_prompt > 0 else 1
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "condition_images": condition_images,
            "vae_images": vae_images,
            "condition_image_sizes": condition_image_sizes,
            "vae_image_sizes": vae_image_sizes,
            "calculated_height": calculated_height,
            "calculated_width": calculated_width,
            "height": height,
            "width": width,
            "num_inference_steps": state.sampling.num_inference_steps or kwargs.get("num_inference_steps", 50),
            "sigmas": state.sampling.sigmas or kwargs.get("sigmas"),
            "guidance_scale": (
                state.sampling.guidance_scale
                if state.sampling.guidance_scale_provided
                else kwargs.get("guidance_scale", 1.0)
            ),
            "num_images_per_prompt": num_images_per_prompt,
            "generator": state.sampling.generator or kwargs.get("generator"),
            "true_cfg_scale": (
                state.sampling.true_cfg_scale
                if state.sampling.true_cfg_scale is not None
                else kwargs.get("true_cfg_scale", 4.0)
            ),
            "max_sequence_length": state.sampling.max_sequence_length
            or kwargs.get("max_sequence_length", self.tokenizer_max_length),
            "latents": kwargs.get("latents"),
            "prompt_embeds": kwargs.get("prompt_embeds"),
            "prompt_embeds_mask": kwargs.get("prompt_embeds_mask"),
            "negative_prompt_embeds": kwargs.get("negative_prompt_embeds"),
            "negative_prompt_embeds_mask": kwargs.get("negative_prompt_embeds_mask"),
            "output_type": state.sampling.output_type or kwargs.get("output_type", "pil"),
            "attention_kwargs": kwargs.get("attention_kwargs"),
            "callback_on_step_end_tensor_inputs": kwargs.get("callback_on_step_end_tensor_inputs"),
        }

    def validation(self, state: DiffusionRequestState) -> DiffusionRequestState:
        args = get_qwen_state(state, "atom_args")
        if args is None:
            args = self._prepare_atom_args(state)
        self.check_inputs(
            args["prompt"],
            args["height"],
            args["width"],
            args["condition_images"],
            args["negative_prompt"],
            args["prompt_embeds"],
            args["negative_prompt_embeds"],
            args["prompt_embeds_mask"],
            args["negative_prompt_embeds_mask"],
            args["callback_on_step_end_tensor_inputs"],
            args["max_sequence_length"],
        )

        self._guidance_scale = args["guidance_scale"]
        self._attention_kwargs = args["attention_kwargs"] or {}
        self._current_timestep = None
        self._interrupt = False

        prompt = args["prompt"]
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = args["prompt_embeds"].shape[0] if args["prompt_embeds"] is not None else 1

        has_neg_prompt = args["negative_prompt"] is not None or (
            args["negative_prompt_embeds"] is not None and args["negative_prompt_embeds_mask"] is not None
        )
        do_true_cfg = args["true_cfg_scale"] > 1 and has_neg_prompt
        self.check_cfg_parallel_validity(args["true_cfg_scale"], has_neg_prompt)
        state.sampling.height = args["height"]
        state.sampling.width = args["width"]
        state.sampling.output_type = args["output_type"]
        set_qwen_state(
            state,
            atom_args=args,
            batch_size=batch_size,
            do_true_cfg=do_true_cfg,
            output_type=args["output_type"],
            attention_kwargs=args["attention_kwargs"] or {},
        )
        return state

    def encoding(self, state: DiffusionRequestState) -> DiffusionRequestState:
        args = get_qwen_state(state, "atom_args")
        do_true_cfg = get_qwen_state(state, "do_true_cfg", False)

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=args["prompt"],
            image=args["condition_images"],
            prompt_embeds=args["prompt_embeds"],
            prompt_embeds_mask=args["prompt_embeds_mask"],
            num_images_per_prompt=args["num_images_per_prompt"],
            max_sequence_length=args["max_sequence_length"],
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=args["negative_prompt"],
                image=args["condition_images"],
                prompt_embeds=args["negative_prompt_embeds"],
                prompt_embeds_mask=args["negative_prompt_embeds_mask"],
                num_images_per_prompt=args["num_images_per_prompt"],
                max_sequence_length=args["max_sequence_length"],
                prompt_name="negative_prompt",
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

        set_qwen_state(
            state,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            txt_seq_lens=prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None,
            negative_txt_seq_lens=(
                negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
            ),
        )
        return state

    def preparation(self, state: DiffusionRequestState) -> DiffusionRequestState:
        args = get_qwen_state(state, "atom_args")
        batch_size = get_qwen_state(state, "batch_size", 1)
        do_true_cfg = get_qwen_state(state, "do_true_cfg", False)
        prompt_embeds = get_qwen_state(state, "prompt_embeds")

        num_channels_latents = qwen_transformer_in_channels(self) // 4
        latents, image_latents = self.prepare_latents(
            args["vae_images"],
            batch_size * args["num_images_per_prompt"],
            num_channels_latents,
            args["height"],
            args["width"],
            prompt_embeds.dtype,
            self.device,
            args["generator"],
            args["latents"],
        )
        img_shapes = [
            [
                (1, args["height"] // self.vae_scale_factor // 2, args["width"] // self.vae_scale_factor // 2),
                *[
                    (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
                    for vae_width, vae_height in args["vae_image_sizes"]
                ],
            ]
        ] * (batch_size * args["num_images_per_prompt"])

        timesteps, _ = self.prepare_timesteps(args["num_inference_steps"], args["sigmas"], latents.shape[1])
        self._num_timesteps = len(timesteps)

        if qwen_transformer_guidance_embeds(self):
            guidance = torch.full([1], args["guidance_scale"], dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        req_scheduler = copy.deepcopy(self.scheduler)
        req_scheduler.set_begin_index(0)
        state.latents = latents
        state.timesteps = timesteps
        state.step_index = 0
        state.scheduler = req_scheduler
        state.sampling.cfg_normalize = True
        self._set_runner_cfg(
            state,
            do_true_cfg=do_true_cfg,
            true_cfg_scale=args["true_cfg_scale"],
            cfg_normalize=True,
        )
        set_qwen_state(
            state,
            guidance=guidance,
            image_latents=image_latents,
            img_shapes=img_shapes,
        )
        return state

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        image: PIL.Image.Image | list[PIL.Image.Image] | torch.Tensor | None = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 1024,
    ) -> DiffusionOutput:
        """Forward pass for image editing with support for multiple images."""
        # TODO: In online mode, sometimes it receives [{"negative_prompt": None}, {...}], so cannot use .get("...", "")
        # TODO: May be some data formatting operations on the API side. Hack for now.
        if len(req.prompts) > 1:
            logger.warning(
                """This model only supports a single prompt, not a batched request.""",
                """Taking only the first image for now.""",
            )
        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
        negative_prompt = None if isinstance(first_prompt, str) else first_prompt.get("negative_prompt")
        if negative_prompt is None:
            logger.warning(
                "negative_prompt is not set. The official Qwen-Image-Edit model "
                "may produce lower-quality results without a negative_prompt. "
                "Qwen official repository recommends to use whitespace string as negative_prompt. "
                "Note: some distilled variants may not be affected by this."
            )

        # Get preprocessed images from request (pre-processing is done in DiffusionEngine)
        if (
            not isinstance(first_prompt, str)
            and "vae_images" in (additional_information := first_prompt.get("additional_information", {}))
            and "condition_images" in additional_information
        ):
            condition_images = additional_information.get("condition_images")
            vae_images = additional_information.get("vae_images")
            condition_image_sizes = additional_information.get("condition_image_sizes")
            vae_image_sizes = additional_information.get("vae_image_sizes")
            calculated_height = additional_information.get("calculated_height")
            calculated_width = additional_information.get("calculated_width")
            height = req.sampling_params.height
            width = req.sampling_params.width
        else:
            # fallback to run pre-processing in pipeline (debug only)
            if image is None:
                raise ValueError("Image is required for QwenImageEditPlusPipeline")

            if not isinstance(image, list):
                image = [image]

            image_size = image[0].size
            calculated_width, calculated_height = calculate_dimensions(VAE_IMAGE_SIZE, image_size[0] / image_size[1])
            height = height or calculated_height
            width = width or calculated_width

            height, width = normalize_min_aligned_size(height, width, self.vae_scale_factor * 2)

            condition_images = []
            vae_images = []
            condition_image_sizes = []
            vae_image_sizes = []

            for img in image:
                image_width, image_height = img.size
                condition_width, condition_height = calculate_dimensions(
                    CONDITION_IMAGE_SIZE, image_width / image_height
                )
                vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
                condition_image_sizes.append((condition_width, condition_height))
                vae_image_sizes.append((vae_width, vae_height))
                condition_images.append(self.image_processor.resize(img, condition_height, condition_width))
                vae_images.append(self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))

        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        sigmas = req.sampling_params.sigmas or sigmas
        max_sequence_length = req.sampling_params.max_sequence_length or max_sequence_length
        generator = req.sampling_params.generator or generator
        true_cfg_scale = (
            req.sampling_params.true_cfg_scale
            if req.sampling_params.true_cfg_scale is not None
            else true_cfg_scale
        )
        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else num_images_per_prompt
        )

        if req.sampling_params.output_type is not None:
            output_type = req.sampling_params.output_type

        # 1. check inputs
        # 2. encode prompts
        # 3. prepare latents and timesteps
        # 4. diffusion process
        # 5. decode latents
        # 6. post-process outputs
        self.check_inputs(
            prompt,
            height,
            width,
            image,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs,
            max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )

        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        self.check_cfg_parallel_validity(true_cfg_scale, has_neg_prompt)

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            image=condition_images,  # Use condition images for prompt encoding
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                image=condition_images,  # Use same condition images for negative prompt encoding
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                prompt_name="negative_prompt",
            )

        num_channels_latents = self.transformer.in_channels // 4
        # random noise latents, and image latents encoded by vae
        latents, image_latents = self.prepare_latents(
            vae_images,  # Use VAE images for latent preparation
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )
        # img_shapes includes shapes for output image and all input images
        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                *[
                    (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
                    for vae_width, vae_height in vae_image_sizes
                ],
            ]
        ] * batch_size

        timesteps, num_inference_steps = self.prepare_timesteps(num_inference_steps, sigmas, latents.shape[1])
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.guidance_embeds:
            guidance = torch.full([1], guidance_scale, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        latents = self.diffuse(
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            latents,
            img_shapes,
            txt_seq_lens,
            negative_txt_seq_lens,
            timesteps,
            do_true_cfg,
            guidance,
            true_cfg_scale,
            image_latents=image_latents,
            cfg_normalize=True,
            additional_transformer_kwargs={
                "return_dict": False,
                "attention_kwargs": self.attention_kwargs,
            },
        )

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return load_qwen_transformer_weights(self, weights)
