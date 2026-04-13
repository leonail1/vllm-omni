# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wan2.2 S2V pipeline wrapper used by vLLM-Omni.

This module adapts Wan2.2's native speech-to-video runtime to the
``DiffusionEngine`` contract that powers Omni offline inference and the
OpenAI-compatible serving layer.
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Iterable
from typing import Any

import PIL.Image
import torch
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import SupportAudioInput, SupportImageInput
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.wan2_2.wan2_2_s2v_audio import (
    DEFAULT_S2V_MAX_AREA,
    choose_s2v_dimensions,
    normalize_audio_input,
    normalize_audio_request_input,
    normalize_image_input,
    normalize_pose_video_input,
)
from vllm_omni.diffusion.models.wan2_2.wan2_2_s2v_transformer import Wan22S2VBackend
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniTextPrompt

logger = logging.getLogger(__name__)


def _resolve_tts_request_args(prompt: dict[str, Any], extra_args: dict[str, Any]) -> dict[str, Any]:
    """Collect normalized TTS arguments for a single prompt.

    Request-scoped ``extra_args`` take precedence over prompt-level
    ``additional_information`` so the API server can override serialized prompt
    metadata when it expands multipart requests.

    Args:
        prompt: Prompt payload that may already contain
            ``additional_information``.
        extra_args: Sampling ``extra_args`` forwarded from the current request.

    Returns:
        A normalized dictionary containing the effective TTS mode flag and the
        prompt/text payloads expected by the backend.
    """
    additional_information = prompt.get("additional_information", {})
    enable_tts = extra_args.get("enable_tts")
    if enable_tts is None:
        enable_tts = additional_information.get("enable_tts", False)
    return {
        "enable_tts": bool(enable_tts),
        "tts_prompt_audio": extra_args.get(
            "tts_prompt_audio",
            additional_information.get("tts_prompt_audio"),
        ),
        "tts_prompt_text": extra_args.get(
            "tts_prompt_text",
            additional_information.get("tts_prompt_text"),
        ),
        "tts_text": extra_args.get(
            "tts_text",
            additional_information.get("tts_text"),
        ),
    }


def _validate_tts_request_args(tts_args: dict[str, Any]) -> None:
    """Validate Wan2.2 S2V TTS arguments before backend execution.

    Args:
        tts_args: Normalized TTS arguments produced by
            :func:`_resolve_tts_request_args`.

    Raises:
        ValueError: If a required TTS field is missing or empty.
        TypeError: If ``tts_prompt_text`` is provided with an unexpected type.
    """
    if not tts_args["enable_tts"]:
        return
    if tts_args["tts_prompt_audio"] is None:
        raise ValueError("Wan2.2 S2V TTS mode requires `extra_args[\"tts_prompt_audio\"]`.")
    if not isinstance(tts_args["tts_text"], str) or not tts_args["tts_text"].strip():
        raise ValueError("Wan2.2 S2V TTS mode requires a non-empty `extra_args[\"tts_text\"]` string.")
    tts_prompt_text = tts_args["tts_prompt_text"]
    if tts_prompt_text is not None and not isinstance(tts_prompt_text, str):
        raise TypeError("Wan2.2 S2V expects `tts_prompt_text` to be a string when provided.")


def get_wan22_s2v_post_process_func(od_config: OmniDiffusionConfig):
    """Create the post-processing hook for Wan2.2 S2V outputs.

    Args:
        od_config: Diffusion model configuration. The argument is accepted to
            match the registry callback contract, although the current
            implementation only needs the model-independent ``VideoProcessor``.

    Returns:
        A callable that converts decoded latent videos into numpy/PIL outputs
        and attaches Wan2.2 S2V's native 16 FPS metadata.
    """
    del od_config
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(video: torch.Tensor, output_type: str = "np"):
        """Convert decoded Wan video tensors into API-friendly payloads.

        Args:
            video: Decoded video tensor in Wan's native output layout.
            output_type: Output format requested by the caller.

        Returns:
            Either the original latent tensor when ``output_type`` is
            ``"latent"`` or a dictionary containing the normalized video plus
            the model's fixed output FPS.
        """
        if output_type == "latent":
            return video
        return {
            "video": video_processor.postprocess_video(video, output_type=output_type),
            "fps": 16,
        }

    return post_process_func


def get_wan22_s2v_pre_process_func(od_config: OmniDiffusionConfig):
    """Create the request pre-processing hook for Wan2.2 S2V.

    The pre-processor validates multimodal inputs, resolves default video
    parameters, and normalizes image/audio references into forms that the
    runtime path can consume consistently.

    Args:
        od_config: Diffusion model configuration used to satisfy the registry
            callback signature.

    Returns:
        A callable that mutates and returns the normalized
        :class:`OmniDiffusionRequest`.
    """
    del od_config

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        """Normalize a Wan2.2 S2V request before execution.

        Args:
            request: Diffusion request produced by the serving or offline
                inference layer.

        Returns:
            The same request object with normalized dimensions, FPS defaults,
            and multimodal payloads.

        Raises:
            ValueError: If the mandatory image/audio inputs are missing.
            TypeError: If a multimodal payload has an unsupported type.
        """
        if request.sampling_params.num_frames <= 1:
            request.sampling_params.num_frames = 81
        if request.sampling_params.fps is None:
            request.sampling_params.fps = 16
        if request.sampling_params.frame_rate is None:
            request.sampling_params.frame_rate = 16.0

        extra_args = request.sampling_params.extra_args or {}

        for idx, prompt in enumerate(request.prompts):
            if isinstance(prompt, str):
                prompt = OmniTextPrompt(prompt=prompt)

            multi_modal_data = prompt.setdefault("multi_modal_data", {})
            prompt.setdefault("additional_information", {})
            tts_args = _resolve_tts_request_args(prompt, extra_args)
            _validate_tts_request_args(tts_args)

            raw_image = multi_modal_data.get("image")
            raw_audio = multi_modal_data.get("audio")
            if raw_image is None:
                raise ValueError(
                    "Wan2.2 S2V requires `multi_modal_data[\"image\"]` to be provided."
                )
            if raw_audio is None and not tts_args["enable_tts"]:
                raise ValueError(
                    "Wan2.2 S2V requires `multi_modal_data[\"audio\"]` to be provided unless TTS mode is enabled."
                )

            image = normalize_image_input(raw_image)
            max_area = extra_args.get("max_area", DEFAULT_S2V_MAX_AREA)
            height, width = choose_s2v_dimensions(
                image,
                request.sampling_params.height,
                request.sampling_params.width,
                max_area=max_area,
            )
            request.sampling_params.height = height
            request.sampling_params.width = width

            multi_modal_data["image"] = image
            multi_modal_data["pose_video"] = normalize_pose_video_input(multi_modal_data.get("pose_video"))

            multi_modal_data["audio"] = (
                None if raw_audio is None else normalize_audio_request_input(raw_audio)
            )

            prompt["additional_information"]["preprocessed_image"] = image
            request.prompts[idx] = prompt

        return request

    return pre_process_func


class Wan22S2VPipeline(
    nn.Module,
    SupportImageInput,
    SupportAudioInput,
    CFGParallelMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
):
    """vLLM-Omni pipeline adapter for Wan2.2 speech-to-video generation."""

    _PROFILER_TARGETS = []

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        """Initialize the Wan2.2 S2V pipeline wrapper.

        Args:
            od_config: Diffusion model configuration for the current worker.
            prefix: Unused compatibility argument kept to match the standard
                pipeline constructor signature.
        """
        super().__init__()
        del prefix
        self.od_config = od_config
        self.device = get_local_device()
        self.weights_sources: list[Any] = []

        dtype = getattr(od_config, "dtype", torch.bfloat16)
        self.backend = Wan22S2VBackend(
            model_path=od_config.model,
            device=self.device,
            dtype=dtype,
            enable_cpu_offload=od_config.enable_cpu_offload,
        )
        self.default_fps = self.backend.default_fps
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Report already-loaded parameters to the generic loader path.

        Wan2.2 S2V loads its native checkpoints eagerly during backend
        construction, so there is no diffusers-compatible weight stream left to
        consume here.

        Args:
            weights: Unused generic weight iterator from the model registry.

        Returns:
            The parameter names that are already materialized on the module.
        """
        del weights
        return {name for name, _ in self.named_parameters()}

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        image: PIL.Image.Image | None = None,
        audio: Any | None = None,
        pose_video: str | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        frame_num: int | None = None,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        """Execute one Wan2.2 S2V request and package the result for Omni.

        Args:
            req: Normalized diffusion request.
            prompt: Optional prompt override supplied by the caller.
            negative_prompt: Optional negative prompt override.
            image: Optional image override. Falls back to ``multi_modal_data``.
            audio: Optional audio override. Falls back to ``multi_modal_data``.
            pose_video: Optional pose-video override.
            height: Optional height override.
            width: Optional width override.
            num_inference_steps: Optional sampling-step override.
            guidance_scale: Optional CFG override.
            frame_num: Optional frame-count override kept for compatibility.
            output_type: Requested output type. The native path always returns
                decoded video and metadata through :class:`DiffusionOutput`.
            generator: Unused compatibility argument.
            **kwargs: Unused compatibility arguments.

        Returns:
            Diffusion output containing decoded video frames, output FPS, and
            optional generated TTS audio metadata.

        Raises:
            ValueError: If the request is malformed or unsupported.
        """
        del output_type, generator, kwargs

        if len(req.prompts) != 1:
            raise ValueError("Wan2.2 S2V only supports a single prompt per request.")

        prompt_data = req.prompts[0]
        if isinstance(prompt_data, str):
            prompt_data = OmniTextPrompt(prompt=prompt_data)
        multi_modal_data = prompt_data.get("multi_modal_data", {})

        if prompt is None:
            prompt = prompt_data.get("prompt")
        if negative_prompt is None:
            negative_prompt = prompt_data.get("negative_prompt")
        if image is None:
            image = multi_modal_data.get("image")
        if audio is None:
            audio = multi_modal_data.get("audio")
        if pose_video is None:
            pose_video = multi_modal_data.get("pose_video")

        extra_args = req.sampling_params.extra_args or {}
        tts_args = _resolve_tts_request_args(prompt_data, extra_args)
        _validate_tts_request_args(tts_args)

        if prompt is None:
            raise ValueError("Wan2.2 S2V requires a text prompt.")
        if image is None:
            raise ValueError("Wan2.2 S2V requires an input image.")
        if audio is None and not tts_args["enable_tts"]:
            raise ValueError("Wan2.2 S2V requires an input audio source unless TTS mode is enabled.")

        image_obj = image.convert("RGB") if isinstance(image, PIL.Image.Image) else normalize_image_input(image)
        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        if height is None or width is None:
            height, width = choose_s2v_dimensions(image_obj, height, width)

        infer_frames = extra_args.get("infer_frames")
        num_repeat = extra_args.get("num_repeat", extra_args.get("num_clip"))
        sample_solver = extra_args.get("solver_name", extra_args.get("sample_solver", "unipc"))
        shift = extra_args.get("shift")
        offload_model = bool(extra_args.get("offload_model", self.od_config.enable_cpu_offload))
        init_first_frame = bool(extra_args.get("init_first_frame", extra_args.get("start_from_ref", False)))

        start_time = time.perf_counter()
        prep_duration = 0.0
        with tempfile.TemporaryDirectory(prefix="wan22_s2v_") as temp_dir:
            prep_start = time.perf_counter()
            audio_path = None if tts_args["enable_tts"] else normalize_audio_input(audio, temp_dir)
            tts_prompt_audio = None
            if tts_args["enable_tts"]:
                tts_prompt_audio = normalize_audio_input(tts_args["tts_prompt_audio"], temp_dir)
            pose_video_path = normalize_pose_video_input(pose_video)
            prep_duration = time.perf_counter() - prep_start

            generate_start = time.perf_counter()
            video, generated_audio_path = self.backend.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_obj,
                audio_path=audio_path,
                enable_tts=tts_args["enable_tts"],
                tts_prompt_audio=tts_prompt_audio,
                tts_prompt_text=tts_args["tts_prompt_text"],
                tts_text=tts_args["tts_text"],
                pose_video=pose_video_path,
                height=height,
                width=width,
                num_frames=req.sampling_params.num_frames or frame_num,
                infer_frames=infer_frames,
                num_inference_steps=req.sampling_params.num_inference_steps or num_inference_steps,
                guidance_scale=(
                    req.sampling_params.guidance_scale
                    if req.sampling_params.guidance_scale_provided
                    else guidance_scale
                ),
                shift=shift,
                sample_solver=sample_solver,
                seed=req.sampling_params.seed if req.sampling_params.seed is not None else -1,
                num_repeat=num_repeat,
                offload_model=offload_model,
                init_first_frame=init_first_frame,
                temp_dir=temp_dir,
            )
            generate_duration = time.perf_counter() - generate_start

        if video.ndim == 4:
            video = video.unsqueeze(0)

        stage_durations = {
            "Wan22S2VPipeline.prepare": prep_duration,
            "Wan22S2VPipeline.generate": generate_duration,
            "Wan22S2VPipeline.total": time.perf_counter() - start_time,
        }
        if getattr(self, "enable_diffusion_pipeline_profiler", False):
            stage_durations.update(self.stage_durations)

        return DiffusionOutput(
            output=video,
            stage_durations=stage_durations,
            custom_output={
                "fps": self.default_fps,
                "num_repeat": num_repeat,
                "audio_path": generated_audio_path,
                "cleanup_audio_path": generated_audio_path is not None,
            },
        )
