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
from vllm_omni.diffusion.models.qwen_image.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage,
)
from vllm_omni.diffusion.models.qwen_image.cfg_parallel import (
    QwenImageCFGParallelMixin,
)
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


# Interface called in diffusion engine
def get_qwen_image_layered_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-processing function for QwenImageLayeredPipeline."""
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])

    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** len(vae_config["temporal_downsample"]) if "temporal_downsample" in vae_config else 8
        latent_channels = vae_config.get("z_dim", 16)

    # QwenImage latents are turned into 2x2 patches and packed.
    # This means the latent width and height has to be divisible
    # by the patch size. So the vae scale factor is multiplied
    # by the patch size to account for this
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    def pre_process_func(
        request: OmniDiffusionRequest,
    ):
        """Pre-process requests for QwenImageLayeredPipeline."""
        for i, prompt in enumerate(request.prompts):
            multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
            raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
            if isinstance(prompt, str):
                prompt = OmniTextPrompt(prompt=prompt)
            if "additional_information" not in prompt:
                prompt["additional_information"] = {}

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

            if isinstance(image, PIL.Image.Image) and image.mode != "RGBA":
                image = image.convert("RGBA")

            # 1. calculate dimensions
            image_size = image.size
            assert request.sampling_params.resolution in [640, 1024], (
                f"resolution must be either 640 or 1024, but got {request.sampling_params.resolution}"
            )
            calculated_width, calculated_height = calculate_dimensions(
                request.sampling_params.resolution * request.sampling_params.resolution, image_size[0] / image_size[1]
            )
            height = calculated_height
            width = calculated_width

            height, width = normalize_min_aligned_size(height, width, vae_scale_factor * 2)

            # Store calculated dimensions in request
            prompt["additional_information"]["calculated_height"] = calculated_height
            prompt["additional_information"]["calculated_width"] = calculated_width
            request.sampling_params.height = height
            request.sampling_params.width = width

            # 2. Preprocess image
            if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == latent_channels):
                image = image_processor.resize(image, calculated_height, calculated_width)
                prompt_image = image
                image = image_processor.preprocess(image, calculated_height, calculated_width)
                image = image.unsqueeze(2)
                # image = image.to(dtype=self.text_encoder.dtype)  # do it later

                # Store preprocessed image and prompt image in request
                prompt["additional_information"]["preprocessed_image"] = image
                prompt["additional_information"]["prompt_image"] = prompt_image
            request.prompts[i] = prompt
        return request

    return pre_process_func


# Copied from diffusers to avoid version coupling.
# Upstream code merged on 2025-12-18.
def calculate_dimensions(target_area, ratio):
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
):
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
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class QwenImageLayeredPipeline(
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

    color_format = "RGBA"

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

        # modules keep same as transformers & diffusers
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

        if owns_transformer:
            transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, QwenImageTransformer2DModel)
            self.transformer = QwenImageTransformer2DModel(od_config=od_config, **transformer_kwargs)
            self.transformer_in_channels = self.transformer.in_channels
            self.transformer_guidance_embeds = self.transformer.guidance_embeds
        else:
            self.transformer = None
            self.transformer_in_channels = int(self._get_tf_config_value("in_channels", 64))
            self.transformer_guidance_embeds = bool(self._get_tf_config_value("guidance_embeds", False))

        # Pipeline configuration & processing parameters
        self.vae_scale_factor = (
            2 ** len(self.vae.temperal_downsample) if self.vae is not None else self.DEFAULT_VAE_SCALE_FACTOR
        )
        self.latent_channels = self.vae.config.z_dim if self.vae is not None else self.DEFAULT_LATENT_CHANNELS
        # QwenImage latents are turned into 2x2 patches and packed.
        # This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied
        # by the patch size to account for this
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.vl_processor = self.processor
        self.tokenizer_max_length = 1024

        self.prompt_template_encode = (
            "<|im_start|>system\nDescribe the image by detailing the color, "
            "shape, size, texture, quantity, text, spatial relationships of the objects and "
            "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        self.prompt_template_encode_start_idx = 34
        self.image_caption_prompt_cn = (
            """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n# """
            """图像标注器\n你是一个专业的图像标注器。请基于输入图像，撰写图注:\n1.
使用自然、描述性的语言撰写图注，不要使用结构化形式或富文本形式。\n2. 通过加入以下内容，丰富图注细节：\n """
            """- 对象的属性：如数量、颜色、形状、大小、位置、材质、状态、动作等\n -
对象间的视觉关系：如空间关系、功能关系、动作关系、从属关系、比较关系、因果关系等\n - 环境细节：例如天气、"""
            """光照、颜色、纹理、气氛等\n - 文字内容：识别图像中清晰可见的文字，不做翻译和解释，用引号在"""
            """图注中强调\n3.
保持真实性与准确性：\n - 不要使用笼统的描述\n -
描述图像中所有可见的信息，但不要加入没有在图像中出现"""
            """的内容\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"""
        )
        self.image_caption_prompt_en = (
            """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"""
            """<|im_start|>user\n# Image Annotator\nYou are a professional
image annotator. Please write an image caption based on the input image:\n1. Write the caption using natural,
descriptive language without structured formats or rich text.\n2. Enrich caption details by including: \n - Object
attributes, such as quantity, color, shape, size, material, state, position, actions, and so on\n - Vision Relations
between objects, such as spatial relations, functional relations, possessive relations, attachment relations, action
relations, comparative relations, causal relations, and so on\n - Environmental details, such as weather, lighting,
colors, textures, atmosphere, and so on\n - Identify the text clearly visible in the image, without translation or
explanation, and highlight it in the caption with quotation marks\n3. Maintain authenticity and accuracy:\n - Avoid
generalizations\n - Describe all visible information in the image, while do not add information not explicitly shown in
the image\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"""
        )
        self.default_sample_size = 128
        self._guidance_scale = 0.0
        self._attention_kwargs = {}
        self._num_timesteps = 0
        self._current_timestep = None
        self._interrupt = False

        self.stage = self.stage_role
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    @classmethod
    def _resolve_stage_role(cls, od_config: OmniDiffusionConfig) -> str:
        role = getattr(od_config, "stage_role", "all")
        if role not in cls._ROLE_COMPONENTS:
            raise ValueError(f"Unsupported QwenImageLayered stage_role: {role}")
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
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
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
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `prompt_embeds_mask` also have to be passed. Make sure to generate"
                " `prompt_embeds_mask` from the same text encoder that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` also have to be passed. Make"
                " sure to generate `negative_prompt_embeds_mask` from the same text encoder that was used to "
                "generate `negative_prompt_embeds`."
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
        prompt: str | list[str] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        max_sequence_length: int | None = None,
        prompt_name: str = "prompt",
    ):
        device = device or self.device
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
        ).to(device)
        # The layered template also appends fixed non-user tokens after the
        # editable text, so use the empty-template tokenized baseline instead of
        # counting everything after prompt_template_encode_start_idx.
        template_tokens = self.tokenizer(
            [template.format("")],
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(device)
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

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask

    def _encode_prompt_impl(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
        prompt_name: str = "prompt",
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """
        device = device or self.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
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
        layers,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (
            batch_size,
            layers + 1,
            num_channels_latents,
            height,
            width,
        )  ### the generated first image is combined image

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
            image_latents = image_latents.permute(0, 2, 1, 3, 4)  # (b, c, f, h, w) -> (b, f, c, h, w)
            image_latents = self._pack_latents(
                image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width, 1
            )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width, layers + 1)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents

    def get_image_caption(self, prompt_image, use_en_prompt=True, device=None):
        if use_en_prompt:
            prompt = self.image_caption_prompt_en
        else:
            prompt = self.image_caption_prompt_cn
        model_inputs = self.vl_processor(
            text=prompt,
            images=prompt_image,
            padding=True,
            return_tensors="pt",
        ).to(device)
        generated_ids = self.text_encoder.generate(**model_inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        output_text = self.vl_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text.strip()

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width, layers):
        latents = latents.view(batch_size, layers, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4, 6)
        latents = latents.reshape(batch_size, layers * (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, layers, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, layers + 1, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 1, 4, 2, 5, 3, 6)

        latents = latents.reshape(batch_size, layers + 1, channels // (2 * 2), height, width)
        latents = latents.permute(0, 2, 1, 3, 4)  # (b, c, f, h, w)

        return latents

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
        if (
            self.text_encoder is None
            or self.tokenizer is None
            or self.processor is None
            or self.scheduler is None
            or self.vae is None
        ):
            raise RuntimeError(
                "QwenImageLayered encode_stage requires scheduler, text_encoder, tokenizer, processor, and vae."
            )

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

        if isinstance(first_prompt, str):
            raise ValueError("QwenImageLayeredPipeline requires preprocessed image information.")
        additional_information = first_prompt.get("additional_information", {})
        if "preprocessed_image" not in additional_information:
            raise ValueError("Preprocessed image is required for QwenImageLayeredPipeline.")

        prompt_image = additional_information.get("prompt_image")
        image = additional_information.get("preprocessed_image")
        image = image.to(dtype=self.text_encoder.dtype)
        calculated_height = additional_information.get("calculated_height")
        calculated_width = additional_information.get("calculated_width")
        height = sampling.height
        width = sampling.width

        layers = sampling.layers if sampling.layers is not None else 4
        max_sequence_length = sampling.max_sequence_length or self.tokenizer_max_length
        cfg_normalize = bool(sampling.cfg_normalize) if sampling.cfg_normalize is not None else False
        use_en_prompt = bool(sampling.use_en_prompt) if sampling.use_en_prompt is not None else False
        num_inference_steps = sampling.num_inference_steps or 50
        sigmas = sampling.sigmas
        generator = sampling.generator
        true_cfg_scale = sampling.true_cfg_scale or 4.0
        guidance_scale = sampling.guidance_scale if sampling.guidance_scale_provided else None
        num_images_per_prompt = sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else 1
        output_type = sampling.output_type or "pil"

        sampling.height = height
        sampling.width = width
        sampling.layers = layers
        sampling.output_type = output_type
        inputs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "prompt_image": prompt_image,
            "image": image,
            "calculated_height": calculated_height,
            "calculated_width": calculated_width,
            "height": height,
            "width": width,
            "layers": layers,
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
            "output_type": output_type,
            "attention_kwargs": None,
            "cfg_normalize": cfg_normalize,
            "use_en_prompt": use_en_prompt,
        }
        state.extra["_qwen_image_layered_stage_inputs"] = inputs
        return inputs

    def check_inputs(self, state: "DiffusionRequestState") -> None:
        # Atom 1: normalize request fields and run the original layered input checks.
        inputs = self._stage_inputs_from_state(state)
        self._check_inputs_impl(
            inputs["prompt"],
            inputs["height"],
            inputs["width"],
            negative_prompt=inputs["negative_prompt"],
            prompt_embeds=inputs["prompt_embeds"],
            negative_prompt_embeds=inputs["negative_prompt_embeds"],
            prompt_embeds_mask=inputs["prompt_embeds_mask"],
            negative_prompt_embeds_mask=inputs["negative_prompt_embeds_mask"],
            max_sequence_length=inputs["max_sequence_length"],
        )

        self._guidance_scale = inputs["guidance_scale"]
        self._attention_kwargs = inputs["attention_kwargs"]
        self._current_timestep = None
        self._interrupt = False

    def encode_prompt(self, state: "DiffusionRequestState") -> None:
        # Atom 2: caption empty prompts and encode positive/negative text.
        inputs = state.extra["_qwen_image_layered_stage_inputs"]
        prompt = inputs["prompt"]
        prompt_embeds = inputs["prompt_embeds"]
        negative_prompt = inputs["negative_prompt"]
        negative_prompt_embeds = inputs["negative_prompt_embeds"]
        negative_prompt_embeds_mask = inputs["negative_prompt_embeds_mask"]
        true_cfg_scale = inputs["true_cfg_scale"]

        if prompt is None or prompt == "" or prompt == " ":
            prompt = self.get_image_caption(
                inputs["prompt_image"],
                use_en_prompt=inputs["use_en_prompt"],
                device=self.device,
            )
            inputs["prompt"] = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        if true_cfg_scale > 1 and not has_neg_prompt:
            logger.warning(
                f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free "
                f"guidance is not enabled since no negative_prompt is provided."
            )
        elif true_cfg_scale <= 1 and has_neg_prompt:
            logger.warning(
                " negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale <= 1"
            )

        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        self.check_cfg_parallel_validity(true_cfg_scale, has_neg_prompt)

        prompt_embeds, prompt_embeds_mask = self._encode_prompt_impl(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=inputs["prompt_embeds_mask"],
            device=self.device,
            num_images_per_prompt=inputs["num_images_per_prompt"],
            max_sequence_length=inputs["max_sequence_length"],
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self._encode_prompt_impl(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=self.device,
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
        state.extra["_qwen_image_layered_batch_size"] = batch_size

    def prepare_latents(self, state: "DiffusionRequestState") -> None:
        # Atom 3: create random layered latents and encode/pack the conditioning image.
        inputs = state.extra["_qwen_image_layered_stage_inputs"]
        batch_size = int(state.extra["_qwen_image_layered_batch_size"])
        num_channels_latents = self.transformer_in_channels // 4
        latents, image_latents = self._prepare_latents_impl(
            inputs["image"],
            batch_size * inputs["num_images_per_prompt"],
            num_channels_latents,
            inputs["height"],
            inputs["width"],
            inputs["layers"],
            state.prompt_embeds.dtype,
            self.device,
            inputs["generator"],
            inputs["latents"],
        )
        state.latents = latents
        state.sampling.image_latent = image_latents
        state.img_shapes = [
            [
                *[
                    (1, inputs["height"] // self.vae_scale_factor // 2, inputs["width"] // self.vae_scale_factor // 2)
                    for _ in range(inputs["layers"] + 1)
                ],
                (
                    1,
                    inputs["calculated_height"] // self.vae_scale_factor // 2,
                    inputs["calculated_width"] // self.vae_scale_factor // 2,
                ),
            ]
        ] * batch_size

        if self.transformer_guidance_embeds and inputs["guidance_scale"] is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        elif self.transformer_guidance_embeds:
            guidance = torch.full([1], inputs["guidance_scale"], device=self.device, dtype=torch.float32)
            state.guidance = guidance.expand(latents.shape[0])
        elif not self.transformer_guidance_embeds and inputs["guidance_scale"] is not None:
            logger.warning(
                f"guidance_scale is passed as {inputs['guidance_scale']}, but ignored since the model is not "
                "guidance-distilled."
            )
            state.guidance = None
        else:
            state.guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

    def _prepare_timesteps_impl(
        self,
        num_inference_steps: int,
        sigmas: list[float] | None,
        image_latents_seq_len: int,
    ) -> tuple[torch.Tensor, int]:
        sigmas = np.linspace(1.0, 0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
        base_seqlen = 256 * 256 / 16 / 16
        mu = (image_latents_seq_len / base_seqlen) ** 0.5
        return retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            self.device,
            sigmas=sigmas,
            mu=mu,
        )

    def prepare_timesteps(self, state: "DiffusionRequestState") -> None:
        # Atom 4: materialize layered timesteps and request-local scheduler state.
        inputs = state.extra["_qwen_image_layered_stage_inputs"]
        image_latents = state.sampling.image_latent
        timesteps, _ = self._prepare_timesteps_impl(
            inputs["num_inference_steps"],
            inputs["sigmas"],
            image_latents.shape[1],
        )
        self._num_timesteps = len(timesteps)
        req_scheduler = copy.deepcopy(self.scheduler)
        req_scheduler.set_begin_index(0)
        state.step_index = 0
        state.scheduler = req_scheduler
        state.timesteps = timesteps
        state.sampling.cfg_normalize = inputs["cfg_normalize"]
        state.extra.pop("_qwen_image_layered_stage_inputs", None)
        state.extra.pop("_qwen_image_layered_batch_size", None)

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
        is_rgb = torch.tensor([0] * len(img_shapes), device=latents.device, dtype=torch.long)
        additional_transformer_kwargs = {
            "return_dict": False,
            "additional_t_cond": is_rgb,
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

        output_slice = latents.size(1) if image_latents is not None else None
        return positive_kwargs, negative_kwargs, output_slice

    def _decode_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        layers: int,
        output_type: str = "pil",
    ) -> DiffusionOutput:
        if output_type == "latent":
            return DiffusionOutput(
                output=latents,
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )

        if self.vae is None:
            raise RuntimeError("QwenImageLayered decode_stage requires VAE loaded.")
        latents = self._unpack_latents(latents, height, width, layers, self.vae_scale_factor)
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

        b, c, f, h, w = latents.shape
        latents = latents[:, :, 1:]
        latents = latents.permute(0, 2, 1, 3, 4).view(-1, c, 1, h, w)
        image = self.vae.decode(latents, return_dict=False)[0]
        image = image.squeeze(2)
        image = self.image_processor.postprocess(image, output_type=output_type)
        images = []
        for bidx in range(b):
            images.append(image[bidx * f : (bidx + 1) * f])

        return DiffusionOutput(
            output=images,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def _predict_noise_impl(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        if self.transformer is None:
            raise RuntimeError("QwenImageLayered denoise_stage requires transformer loaded.")
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
            raise RuntimeError("QwenImageLayered denoise_stage requires transformer loaded.")

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
        layers = state.sampling.layers if state.sampling.layers is not None else 4
        output_type = state.sampling.output_type or "pil"
        return self._decode_latents(state.latents, height, width, layers, output_type)

    def rehydrate_stage_state(self, state: "DiffusionRequestState") -> "DiffusionRequestState":
        # Transport hook: recreate the scheduler object in the receiving role.
        if self.scheduler is None:
            return state
        if state.scheduler is not None or state.timesteps is None or state.latents is None:
            return state
        image_latents = state.sampling.image_latent
        timesteps = state.timesteps
        self._prepare_timesteps_impl(
            state.total_steps,
            state.sampling.sigmas,
            image_latents.shape[1],
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
