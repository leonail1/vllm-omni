# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, cast

import numpy as np
import soundfile as sf
from fastapi import HTTPException
from PIL import Image
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.protocol.videos import (
    DEFAULT_FPS,
    VideoData,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from vllm_omni.entrypoints.openai.utils import get_stage_type, parse_lora_request
from vllm_omni.entrypoints.openai.video_api_utils import _encode_video_bytes, encode_video_base64
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniSamplingParams, OmniTextPrompt

logger = init_logger(__name__)


@dataclass
class ReferenceImage:
    data: Image.Image


@dataclass
class ReferenceAudio:
    data: np.ndarray
    sample_rate: int

    @property
    def payload(self) -> tuple[np.ndarray, int]:
        return self.data, self.sample_rate


@dataclass
class VideoGenerationArtifacts:
    videos: list[Any]
    audios: list[Any | None]
    audio_sample_rate: int
    output_fps: int
    stage_durations: dict[str, float]
    peak_memory_mb: float


class OmniOpenAIServingVideo:
    def __init__(
        self,
        engine_client: EngineClient,
        model_name: str | None = None,
        stage_configs: list[Any] | None = None,
    ) -> None:
        self._engine_client = engine_client
        self._model_name = model_name
        self._stage_configs = stage_configs

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @property
    def stage_configs(self) -> list[Any] | None:
        return self._stage_configs

    def set_stage_configs_if_missing(self, stage_configs: list[Any] | None) -> None:
        if self._stage_configs is None and stage_configs is not None:
            self._stage_configs = stage_configs

    @classmethod
    def for_diffusion(
        cls,
        diffusion_engine: EngineClient,
        model_name: str,
        stage_configs: list[Any] | None = None,
    ) -> OmniOpenAIServingVideo:
        return cls(
            diffusion_engine,
            model_name=model_name,
            stage_configs=stage_configs,
        )

    async def _run_and_extract(
        self,
        request: VideoGenerationRequest,
        reference_id: str,
        *,
        reference_image: ReferenceImage | None = None,
        reference_audio: ReferenceAudio | None = None,
        tts_prompt_audio: ReferenceAudio | None = None,
    ) -> VideoGenerationArtifacts:
        prompt: OmniTextPrompt = OmniTextPrompt(prompt=request.prompt)
        if request.negative_prompt is not None:
            prompt["negative_prompt"] = request.negative_prompt

        gen_params = OmniDiffusionSamplingParams()

        input_image = None if reference_image is None else reference_image.data
        default_request_fps = self._resolve_default_request_fps(request)
        vp = request.resolve_video_params(default_fps=default_request_fps)
        if input_image is not None and vp.width is not None and vp.height is not None:
            target_size = (vp.width, vp.height)
            if input_image.size != target_size:
                input_image = input_image.resize(target_size, Image.Resampling.LANCZOS)

        multi_modal_data: dict[str, Any] = {}
        if input_image is not None:
            multi_modal_data["image"] = input_image
        if reference_audio is not None:
            multi_modal_data["audio"] = reference_audio.payload
        if multi_modal_data:
            prompt["multi_modal_data"] = multi_modal_data

        if vp.width is not None and vp.height is not None:
            gen_params.width = vp.width
            gen_params.height = vp.height
        if vp.num_frames is not None:
            gen_params.num_frames = vp.num_frames
        if vp.fps is not None:
            gen_params.fps = vp.fps
            gen_params.frame_rate = float(vp.fps)

        if request.num_inference_steps is not None:
            gen_params.num_inference_steps = request.num_inference_steps
        if request.guidance_scale is not None:
            gen_params.guidance_scale = request.guidance_scale
        if request.guidance_scale_2 is not None:
            gen_params.guidance_scale_2 = request.guidance_scale_2
        if request.true_cfg_scale is not None:
            gen_params.true_cfg_scale = request.true_cfg_scale
        if request.seed is not None:
            gen_params.seed = request.seed
        if request.boundary_ratio is not None:
            gen_params.boundary_ratio = request.boundary_ratio

        logger.info(
            "Boundary ratio parse: request=%s gen_params=%s",
            request.boundary_ratio,
            gen_params.boundary_ratio,
        )
        if request.flow_shift is not None:
            gen_params.extra_args["flow_shift"] = request.flow_shift

        if request.extra_params is not None:
            if not isinstance(request.extra_params, dict):
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail="extra_params must be a JSON object/dict.",
                )
            gen_params.extra_args.update(request.extra_params)
            logger.info("Applied extra_params: %s", request.extra_params)

        enable_tts = self._resolve_enable_tts(request)
        if enable_tts:
            gen_params.extra_args["enable_tts"] = True
        if tts_prompt_audio is not None:
            gen_params.extra_args["tts_prompt_audio"] = tts_prompt_audio.payload
        elif request.tts_prompt_audio is not None:
            gen_params.extra_args["tts_prompt_audio"] = request.tts_prompt_audio
        if request.tts_prompt_text is not None:
            gen_params.extra_args["tts_prompt_text"] = request.tts_prompt_text
        if request.tts_text is not None:
            gen_params.extra_args["tts_text"] = request.tts_text

        self._apply_lora(request.lora, gen_params)

        logger.info(
            "Video sampling params: steps=%s guidance=%s guidance_2=%s seed=%s enable_tts=%s",
            gen_params.num_inference_steps,
            gen_params.guidance_scale,
            gen_params.guidance_scale_2,
            gen_params.seed,
            enable_tts,
        )

        result = await self._run_generation(prompt, gen_params, reference_id)
        videos = self._extract_video_outputs(result)
        audios, extracted_audio_sample_rate = self._extract_audio_outputs(result, expected_count=len(videos))

        if reference_audio is not None and not enable_tts:
            audios = [audio if audio is not None else reference_audio.data for audio in audios]
            if extracted_audio_sample_rate is None:
                extracted_audio_sample_rate = reference_audio.sample_rate

        audio_sample_rate = extracted_audio_sample_rate or self._resolve_audio_sample_rate(result)
        output_fps = vp.fps or self._resolve_fps(result) or 24
        return VideoGenerationArtifacts(
            videos=videos,
            audios=audios,
            audio_sample_rate=audio_sample_rate,
            output_fps=output_fps,
            stage_durations=self._extract_stage_durations(result),
            peak_memory_mb=self._extract_peak_memory_mb(result),
        )

    async def generate_videos(
        self,
        request: VideoGenerationRequest,
        reference_id: str,
        *,
        reference_image: ReferenceImage | None = None,
        reference_audio: ReferenceAudio | None = None,
        tts_prompt_audio: ReferenceAudio | None = None,
    ) -> VideoGenerationResponse:
        artifacts = await self._run_and_extract(
            request,
            reference_id,
            reference_image=reference_image,
            reference_audio=reference_audio,
            tts_prompt_audio=tts_prompt_audio,
        )
        _t_encode_start = time.perf_counter()
        video_data = [
            VideoData(
                b64_json=(
                    encode_video_base64(video, fps=artifacts.output_fps)
                    if artifacts.audios[idx] is None
                    else encode_video_base64(
                        video,
                        fps=artifacts.output_fps,
                        audio=artifacts.audios[idx],
                        audio_sample_rate=artifacts.audio_sample_rate,
                    )
                )
            )
            for idx, video in enumerate(artifacts.videos)
        ]
        _t_encode_ms = (time.perf_counter() - _t_encode_start) * 1000
        logger.info("Video response encoding (MP4+base64): %.2f ms", _t_encode_ms)
        return VideoGenerationResponse(
            created=int(time.time()),
            data=video_data,
            stage_durations=artifacts.stage_durations,
            peak_memory_mb=artifacts.peak_memory_mb,
        )

    async def generate_video_bytes(
        self,
        request: VideoGenerationRequest,
        reference_id: str,
        *,
        reference_image: ReferenceImage | None = None,
        reference_audio: ReferenceAudio | None = None,
        tts_prompt_audio: ReferenceAudio | None = None,
    ) -> tuple[bytes, dict[str, float], float]:
        artifacts = await self._run_and_extract(
            request,
            reference_id,
            reference_image=reference_image,
            reference_audio=reference_audio,
            tts_prompt_audio=tts_prompt_audio,
        )
        if len(artifacts.videos) > 1:
            logger.warning(
                "Video request %s generated %d outputs; returning only the first.",
                reference_id,
                len(artifacts.videos),
            )
        audio = artifacts.audios[0]
        _t_encode_start = time.perf_counter()
        video_bytes = _encode_video_bytes(
            artifacts.videos[0],
            fps=artifacts.output_fps,
            **({"audio": audio, "audio_sample_rate": artifacts.audio_sample_rate} if audio is not None else {}),
        )
        _t_encode_ms = (time.perf_counter() - _t_encode_start) * 1000
        logger.info("Video response encoding (MP4 bytes): %.2f ms", _t_encode_ms)
        return video_bytes, artifacts.stage_durations, artifacts.peak_memory_mb

    @staticmethod
    def _resolve_enable_tts(request: VideoGenerationRequest) -> bool:
        if request.enable_tts is not None:
            return bool(request.enable_tts)
        return any(
            value is not None
            for value in (
                request.tts_prompt_audio,
                request.tts_prompt_text,
                request.tts_text,
            )
        )

    def _resolve_default_request_fps(self, request: VideoGenerationRequest) -> int:
        if self._is_wan22_s2v_model(request):
            return 16
        return DEFAULT_FPS

    def _is_wan22_s2v_model(self, request: VideoGenerationRequest) -> bool:
        engine_model_config = getattr(self._engine_client, "model_config", None)
        hf_config = getattr(engine_model_config, "hf_config", None)

        model_type = getattr(hf_config, "model_type", None)
        if isinstance(model_type, str) and model_type.lower() == "s2v":
            return True

        architectures = getattr(hf_config, "architectures", None)
        if isinstance(architectures, (list, tuple)) and any(
            isinstance(arch, str) and arch in {"WanS2VPipeline", "WanModel_S2V"}
            for arch in architectures
        ):
            return True

        model_candidates = [request.model, self._model_name, getattr(engine_model_config, "model", None)]
        for candidate in model_candidates:
            if isinstance(candidate, str) and "wan2.2-s2v" in candidate.lower():
                return True

        return False

    @staticmethod
    def _apply_lora(lora_body: Any, gen_params: OmniDiffusionSamplingParams) -> None:
        try:
            lora_request, lora_scale = parse_lora_request(lora_body)
        except ValueError as e:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=str(e),
            ) from e

        if lora_request is None:
            return

        gen_params.lora_request = lora_request
        if lora_scale is not None:
            gen_params.lora_scale = lora_scale

    async def _run_generation(
        self,
        prompt: OmniTextPrompt,
        gen_params: OmniDiffusionSamplingParams,
        request_id: str,
    ) -> Any:
        stage_configs = self._stage_configs or getattr(self._engine_client, "stage_configs", None)

        if not stage_configs:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                detail="Stage configs not found. Start server with an omni diffusion model.",
            )

        for stage in stage_configs:
            stage_type = get_stage_type(stage)
            if stage_type != "diffusion":
                raise HTTPException(
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                    detail=f"Video generation only supports diffusion stages, found '{stage_type}' stage.",
                )

        engine_client = cast(AsyncOmni, self._engine_client)
        sampling_params_list: list[OmniSamplingParams] = [gen_params for _ in stage_configs]

        result = None
        async for output in engine_client.generate(
            prompt=prompt,
            request_id=request_id,
            sampling_params_list=sampling_params_list,
        ):
            result = output

        if result is None:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail="No output generated from video generation pipeline.",
            )
        return result

    @staticmethod
    def _normalize_video_outputs(videos: Any) -> list[Any]:
        if videos is None:
            return []
        if hasattr(videos, "ndim") and videos.ndim == 5:
            return [videos[i] for i in range(videos.shape[0])]
        if isinstance(videos, list):
            if not videos:
                return []
            first = videos[0]
            if hasattr(first, "ndim") and first.ndim == 5:
                flattened: list[Any] = []
                for item in videos:
                    if hasattr(item, "ndim") and item.ndim == 5:
                        flattened.extend([item[i] for i in range(item.shape[0])])
                    else:
                        flattened.append(item)
                return flattened
            if isinstance(first, list):
                return videos
            if hasattr(first, "ndim") and first.ndim == 3:
                return [videos]
            if isinstance(first, Image.Image):
                return [videos]
            return videos
        return [videos]

    def _extract_video_outputs(self, result: Any) -> list[Any]:
        videos = None
        if hasattr(result, "images") and result.images:
            videos = result.images
        elif hasattr(result, "request_output"):
            request_output = result.request_output
            if isinstance(request_output, dict) and request_output.get("images"):
                videos = request_output["images"]
            elif hasattr(request_output, "images") and request_output.images:
                videos = request_output.images
            elif hasattr(request_output, "multimodal_output") and request_output.multimodal_output:
                videos = request_output.multimodal_output.get("video")
        if videos is None and hasattr(result, "multimodal_output") and result.multimodal_output:
            videos = result.multimodal_output.get("video")

        normalized = self._normalize_video_outputs(videos)
        if not normalized:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail="No video outputs found in generation result.",
            )
        return normalized

    @classmethod
    def _extract_audio_outputs(cls, result: Any, expected_count: int) -> tuple[list[Any | None], int | None]:
        audio = None
        if hasattr(result, "multimodal_output") and result.multimodal_output:
            audio = result.multimodal_output.get("audio")
        elif hasattr(result, "request_output"):
            request_output = result.request_output
            if isinstance(request_output, dict) and request_output.get("multimodal_output"):
                mm_output = request_output.get("multimodal_output") or {}
                audio = mm_output.get("audio")
            elif hasattr(request_output, "multimodal_output") and request_output.multimodal_output:
                audio = request_output.multimodal_output.get("audio")

        sample_rate = cls._extract_audio_sample_rate_from_result(result)
        if audio is not None:
            return cls._normalize_audio_outputs(audio, expected_count), sample_rate

        custom_output = cls._extract_custom_output(result)
        audio_path = custom_output.get("audio_path") if isinstance(custom_output, dict) else None
        if audio_path is None:
            return [None] * expected_count, sample_rate

        cleanup_audio_path = bool(custom_output.get("cleanup_audio_path")) if isinstance(custom_output, dict) else False
        audio_payload, path_sample_rate = cls._load_audio_from_path(audio_path, cleanup_path=cleanup_audio_path)
        if expected_count == 1:
            return [audio_payload], path_sample_rate
        return [audio_payload for _ in range(expected_count)], path_sample_rate

    @staticmethod
    def _normalize_audio_outputs(audio: Any, expected_count: int) -> list[Any | None]:
        if isinstance(audio, (list, tuple)):
            if len(audio) == expected_count and any(hasattr(item, "shape") or hasattr(item, "ndim") for item in audio):
                return list(audio)
            if expected_count == 1:
                return [audio]

        if hasattr(audio, "ndim") and getattr(audio, "ndim", None) is not None and audio.ndim > 1:
            first_dim = getattr(audio, "shape", [0])[0]
            if first_dim == expected_count:
                return [audio[i] for i in range(expected_count)]

        if expected_count == 1:
            return [audio]

        return [audio] + [None] * max(expected_count - 1, 0)

    @staticmethod
    def _extract_custom_output(result: Any) -> dict[str, Any]:
        custom_output = getattr(result, "custom_output", None)
        if isinstance(custom_output, dict):
            return custom_output

        request_output = getattr(result, "request_output", None)
        if isinstance(request_output, dict):
            maybe_custom = request_output.get("custom_output") or {}
            if isinstance(maybe_custom, dict):
                return maybe_custom
        elif hasattr(request_output, "custom_output"):
            maybe_custom = getattr(request_output, "custom_output", None)
            if isinstance(maybe_custom, dict):
                return maybe_custom

        return {}

    @staticmethod
    def _load_audio_from_path(audio_path: str, *, cleanup_path: bool = False) -> tuple[np.ndarray, int]:
        try:
            audio_np, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        except (RuntimeError, OSError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail=f"Failed to read generated audio file: {audio_path}",
            ) from exc

        audio_np = np.asarray(audio_np, dtype=np.float32)
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=-1)
        audio_np = np.squeeze(audio_np)
        if audio_np.ndim == 0 or audio_np.size == 0:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail=f"Generated audio file is empty: {audio_path}",
            )
        if cleanup_path:
            try:
                os.remove(audio_path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("Failed to remove temporary generated audio file %s: %s", audio_path, exc)
        return audio_np.astype(np.float32, copy=False), int(sample_rate)

    def _resolve_audio_sample_rate(self, result: Any) -> int:
        result_sample_rate = self._extract_audio_sample_rate_from_result(result)
        if result_sample_rate is not None:
            return result_sample_rate

        model_config = getattr(self._engine_client, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        config_sample_rate = self._extract_audio_sample_rate_from_config(hf_config)
        if config_sample_rate is not None:
            return config_sample_rate

        return 24000

    @staticmethod
    def _resolve_fps(result: Any) -> int | None:
        multimodal_output = getattr(result, "multimodal_output", None)
        if isinstance(multimodal_output, dict):
            fps = multimodal_output.get("fps")
            if fps is not None:
                try:
                    fps_val = fps.item() if hasattr(fps, "item") else int(fps)
                    if fps_val > 0:
                        return fps_val
                except (TypeError, ValueError):
                    pass

        request_output = getattr(result, "request_output", None)
        if isinstance(request_output, dict):
            mm = request_output.get("multimodal_output") or {}
            if isinstance(mm, dict):
                fps = mm.get("fps")
                if fps is not None:
                    try:
                        fps_val = fps.item() if hasattr(fps, "item") else int(fps)
                        if fps_val > 0:
                            return fps_val
                    except (TypeError, ValueError):
                        pass
        elif hasattr(request_output, "multimodal_output"):
            mm = getattr(request_output, "multimodal_output", None)
            if isinstance(mm, dict):
                fps = mm.get("fps")
                if fps is not None:
                    try:
                        fps_val = fps.item() if hasattr(fps, "item") else int(fps)
                        if fps_val > 0:
                            return fps_val
                    except (TypeError, ValueError):
                        pass

        return None

    @classmethod
    def _extract_audio_sample_rate_from_result(cls, result: Any) -> int | None:
        multimodal_output = getattr(result, "multimodal_output", None)
        if isinstance(multimodal_output, dict):
            sample_rate = cls._coerce_audio_sample_rate(
                multimodal_output.get("audio_sample_rate")
                or multimodal_output.get("sample_rate")
                or multimodal_output.get("sampling_rate")
                or multimodal_output.get("sr")
            )
            if sample_rate is not None:
                return sample_rate

        request_output = getattr(result, "request_output", None)
        if isinstance(request_output, dict):
            multimodal_output = request_output.get("multimodal_output") or {}
            if isinstance(multimodal_output, dict):
                return cls._coerce_audio_sample_rate(
                    multimodal_output.get("audio_sample_rate")
                    or multimodal_output.get("sample_rate")
                    or multimodal_output.get("sampling_rate")
                    or multimodal_output.get("sr")
                )
        elif hasattr(request_output, "multimodal_output"):
            multimodal_output = getattr(request_output, "multimodal_output", None)
            if isinstance(multimodal_output, dict):
                return cls._coerce_audio_sample_rate(
                    multimodal_output.get("audio_sample_rate")
                    or multimodal_output.get("sample_rate")
                    or multimodal_output.get("sampling_rate")
                    or multimodal_output.get("sr")
                )

        return None

    @classmethod
    def _extract_audio_sample_rate_from_config(cls, config: Any) -> int | None:
        if config is None:
            return None

        for attr_name in ("output_sampling_rate", "audio_sample_rate", "sample_rate", "sampling_rate"):
            raw_value = config.get(attr_name) if isinstance(config, dict) else getattr(config, attr_name, None)
            sample_rate = cls._coerce_audio_sample_rate(raw_value)
            if sample_rate is not None:
                return sample_rate

        for component_name in ("vocoder", "audio_vae"):
            component = (
                config.get(component_name) if isinstance(config, dict) else getattr(config, component_name, None)
            )
            if component is None:
                continue

            sample_rate = cls._extract_audio_sample_rate_from_config(component)
            if sample_rate is not None:
                return sample_rate

            component_config = (
                component.get("config") if isinstance(component, dict) else getattr(component, "config", None)
            )
            sample_rate = cls._extract_audio_sample_rate_from_config(component_config)
            if sample_rate is not None:
                return sample_rate

        return None

    @staticmethod
    def _coerce_audio_sample_rate(value: Any) -> int | None:
        if value is None:
            return None

        try:
            sample_rate = value.item() if hasattr(value, "item") else value
            sample_rate = int(sample_rate)
        except (TypeError, ValueError):
            return None

        return sample_rate if sample_rate > 0 else None

    @staticmethod
    def _extract_stage_durations(result: Any) -> dict[str, float]:
        stage_durations = getattr(result, "stage_durations", None)
        return stage_durations if isinstance(stage_durations, dict) else {}

    @staticmethod
    def _extract_peak_memory_mb(result: Any) -> float:
        peak_memory_mb = getattr(result, "peak_memory_mb", 0.0)
        try:
            return float(peak_memory_mb or 0.0)
        except (TypeError, ValueError):
            return 0.0
