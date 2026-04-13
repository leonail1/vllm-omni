# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Wan2.2 S2V runtime adapter for vLLM-Omni."""

from __future__ import annotations

import copy
import gc
import logging
import math
import os
import random
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

_TTS_WORKDIR_ENV_VAR = "VLLM_OMNI_WAN22_S2V_TTS_WORKDIR"
_COSYVOICE_LOAD_LOCK = threading.Lock()


def _resolve_tts_workdir(model_path: str) -> Path:
    """Resolve the working directory used by Wan/CosyVoice TTS assets.

    Args:
        model_path: Local path to the Wan2.2 S2V checkpoint directory.

    Returns:
        A writable directory where Wan's TTS helper code can place downloaded
        assets and transient state.
    """
    override = os.environ.get(_TTS_WORKDIR_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()

    model_parent = Path(model_path).resolve().parent
    default_workdir = model_parent / "wan22_s2v_tts_assets"
    if model_parent.exists() and os.access(model_parent, os.W_OK):
        return default_workdir

    fallback_workdir = Path(tempfile.gettempdir()).resolve() / "wan22_s2v_tts_assets"
    logger.warning(
        "Wan2.2 S2V model parent is not writable; falling back to temporary TTS workdir %s. "
        "Set %s to override this path.",
        fallback_workdir,
        _TTS_WORKDIR_ENV_VAR,
    )
    return fallback_workdir


def import_wan_runtime() -> tuple[Any, Any]:
    """Import the upstream Wan runtime from the active Python environment.

    vLLM-Omni intentionally treats Wan2.2 as an external dependency instead of
    reaching into a sibling source checkout such as ``upstream_refs/Wan2.2``.
    This keeps the two repositories independent and makes wheel/sdist installs
    behave the same way as source checkouts.
    """

    try:
        import wan  # type: ignore[import-not-found]
        from wan.configs import WAN_CONFIGS  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "wan"
        raise RuntimeError(
            "Wan2.2 S2V requires the upstream `wan` Python package and its "
            "dependencies to be installed in the current environment. "
            "Install it separately, for example with `pip install /path/to/Wan2.2`. "
            f"Missing import: {missing_name}."
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            "Wan2.2 S2V requires the upstream `wan` Python package to be "
            "installed in the current environment. Install it separately, for "
            "example with `pip install /path/to/Wan2.2`."
        ) from exc

    return wan, WAN_CONFIGS


def patch_native_wan_sdpa_fallback() -> bool:
    """Patch Wan's S2V attention call sites when flash-attn is unavailable."""

    import wan.modules.attention as attention_mod  # type: ignore[import-not-found]

    if attention_mod.FLASH_ATTN_2_AVAILABLE or attention_mod.FLASH_ATTN_3_AVAILABLE:
        return False
    if getattr(attention_mod, "_vllm_omni_sdpa_patch_applied", False):
        return True

    import wan.modules.model as model_mod  # type: ignore[import-not-found]
    import wan.modules.s2v.model_s2v as s2v_model_mod  # type: ignore[import-not-found]
    import wan.modules.s2v.motioner as motioner_mod  # type: ignore[import-not-found]

    def sdpa_flash_attention(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
    ):
        return attention_mod.attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            fa_version=version,
        )

    model_mod.flash_attention = sdpa_flash_attention
    s2v_model_mod.flash_attention = sdpa_flash_attention
    motioner_mod.flash_attention = sdpa_flash_attention
    attention_mod._vllm_omni_sdpa_patch_applied = True
    logger.warning(
        "flash-attn is not available; Wan2.2 S2V is falling back to torch SDPA. "
        "This is functionally valid but can be significantly slower."
    )
    return True


def stabilize_native_wan_audio_encoder(pipeline: Any) -> None:
    """Keep the upstream wav2vec2 encoder in fp32.

    vLLM loads custom pipelines under a global default dtype context. Without
    resetting the upstream audio encoder here, transformers may materialize the
    wav2vec2 weights directly in bf16, while upstream audio preprocessing still
    feeds float32 input tensors. That mismatch only shows up in the Omni worker
    path and breaks real requests during audio feature extraction.
    """

    audio_encoder = getattr(pipeline, "audio_encoder", None)
    if audio_encoder is None or getattr(audio_encoder, "model", None) is None:
        raise RuntimeError("Wan2.2 S2V audio encoder is missing from the native pipeline.")

    audio_encoder.model = audio_encoder.model.to(dtype=torch.float32)
    audio_encoder.model.eval().requires_grad_(False)


class Wan22S2VBackend:
    """Thin adapter over Wan2.2's native S2V inference stack.

    Phase 1 focuses on a functional vLLM-Omni integration and keeps the
    upstream tensor path intact instead of re-implementing the full S2V DiT.
    """

    def __init__(
        self,
        *,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        enable_cpu_offload: bool = False,
    ) -> None:
        """Initialize the native Wan2.2 S2V backend.

        Args:
            model_path: Local path to the Wan2.2 S2V checkpoint root.
            device: CUDA device used for inference.
            dtype: Parameter dtype used for Wan's major modules.
            enable_cpu_offload: Whether the upstream runtime should initialize
                large modules on CPU and offload aggressively during inference.

        Raises:
            RuntimeError: If Wan is not installed or CUDA is unavailable.
        """
        wan, WAN_CONFIGS = import_wan_runtime()

        patch_native_wan_sdpa_fallback()

        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.enable_cpu_offload = enable_cpu_offload

        self.config = copy.deepcopy(WAN_CONFIGS["s2v-14B"])
        self.config.param_dtype = dtype
        self.config.t5_dtype = dtype

        if device.type != "cuda":
            raise RuntimeError("Wan2.2 S2V currently requires a CUDA device.")

        previous_default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            self.pipeline = wan.WanS2V(
                config=self.config,
                checkpoint_dir=model_path,
                device_id=device.index or 0,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_sp=False,
                t5_cpu=False,
                init_on_cpu=enable_cpu_offload,
                convert_model_dtype=True,
            )
        finally:
            torch.set_default_dtype(previous_default_dtype)
        stabilize_native_wan_audio_encoder(self.pipeline)

        self.default_num_frames = int(self.config.frame_num)
        self.default_infer_frames = self.default_num_frames - 1
        self.default_guidance_scale = float(self.config.sample_guide_scale)
        self.default_shift = float(self.config.sample_shift)
        self.default_num_inference_steps = int(self.config.sample_steps)
        self.default_fps = int(self.config.sample_fps)
        self.tts_workdir = _resolve_tts_workdir(self.model_path)

    @property
    def motion_frames(self) -> int:
        """Return the fixed overlap window used by Wan's motion encoder."""
        return int(self.pipeline.motion_frames)

    def resolve_num_frames(self, requested_num_frames: int | None, infer_frames: int | None) -> tuple[int, int]:
        """Resolve API-level frame counts into Wan-native clip lengths.

        Wan expects total output lengths of the form ``4n + 1`` and internal
        inference windows of the form ``4n``. This helper enforces those
        constraints while preserving explicit user intent where possible.

        Args:
            requested_num_frames: Total frame count requested by the caller.
            infer_frames: Optional explicit Wan-native inference length.

        Returns:
            A tuple ``(num_frames, infer_frames)`` compatible with the native
            S2V pipeline.
        """
        if infer_frames is not None:
            infer_frames = max(4, int(infer_frames))
            infer_frames = infer_frames // 4 * 4
            return infer_frames + 1, infer_frames

        if requested_num_frames is None or requested_num_frames <= 1:
            requested_num_frames = self.default_num_frames

        requested_num_frames = max(5, int(requested_num_frames))
        if requested_num_frames % 4 != 1:
            requested_num_frames = requested_num_frames // 4 * 4 + 1
        return requested_num_frames, requested_num_frames - 1

    def _build_scheduler(self, sample_solver: str, sampling_steps: int, shift: float):
        """Construct the upstream Wan scheduler for the current request.

        Args:
            sample_solver: Solver name requested by the caller.
            sampling_steps: Number of denoising steps.
            shift: Flow-matching shift value forwarded to the scheduler.

        Returns:
            A tuple of ``(scheduler, timesteps)`` ready for the denoising loop.

        Raises:
            NotImplementedError: If the requested solver is unsupported.
        """
        from wan.utils.fm_solvers import (  # type: ignore[import-not-found]
            FlowDPMSolverMultistepScheduler,
            get_sampling_sigmas,
            retrieve_timesteps,
        )
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # type: ignore[import-not-found]

        if sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.pipeline.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.pipeline.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
        else:
            raise NotImplementedError(f"Unsupported Wan2.2 S2V solver: {sample_solver}")

        return sample_scheduler, timesteps

    def _patch_cosyvoice_audio_io(self) -> None:
        """Patch CosyVoice audio loading to accept in-memory tensors/arrays.

        Wan's TTS path imports ``load_wav`` from multiple CosyVoice modules.
        This helper rewires both call sites to the same normalization routine so
        API requests can pass uploaded audio without creating extra wrapper code
        inside the upstream repository.
        """
        import cosyvoice.cli.frontend as frontend_mod  # type: ignore[import-not-found]
        import cosyvoice.utils.file_utils as file_utils_mod  # type: ignore[import-not-found]

        if (
            getattr(file_utils_mod, "_vllm_omni_load_wav_patch_applied", False)
            and getattr(frontend_mod, "_vllm_omni_load_wav_patch_applied", False)
        ):
            return

        file_utils_mod.load_wav = self._load_tts_prompt_audio
        frontend_mod.load_wav = self._load_tts_prompt_audio
        file_utils_mod._vllm_omni_load_wav_patch_applied = True
        frontend_mod._vllm_omni_load_wav_patch_applied = True

    def _ensure_tts_loaded(self) -> None:
        """Load Wan/CosyVoice TTS assets exactly once per backend process.

        The upstream implementation resolves some assets relative to the current
        working directory, so initialization is serialized behind a global lock.
        The process-wide cwd switch cannot be removed until upstream exposes a
        dedicated workdir parameter.
        """
        with _COSYVOICE_LOAD_LOCK:
            if not hasattr(self.pipeline, "cosyvoice"):
                self.tts_workdir.mkdir(parents=True, exist_ok=True)
                current_dir = os.getcwd()
                try:
                    logger.warning(
                        "Wan2.2 S2V TTS initialization temporarily switches the process working directory to %s "
                        "because upstream CosyVoice resolves assets relative to cwd. The load path is serialized, "
                        "but other threads in the same process can still observe this transient cwd change.",
                        self.tts_workdir,
                    )
                    os.chdir(self.tts_workdir)
                    self.pipeline.load_tts()
                finally:
                    os.chdir(current_dir)

            self._patch_cosyvoice_audio_io()

    @staticmethod
    def _load_tts_prompt_audio(
        prompt_audio_path: str | os.PathLike[str] | torch.Tensor | np.ndarray,
        target_sr: int = 16000,
        min_sr: int = 16000,
    ) -> torch.Tensor:
        """Load and normalize prompt audio for CosyVoice inference.

        Args:
            prompt_audio_path: Filesystem path or in-memory waveform payload.
            target_sr: Sampling rate expected by CosyVoice.
            min_sr: Minimum accepted source sampling rate before resampling.

        Returns:
            A ``[1, num_samples]`` float32 tensor at ``target_sr``.

        Raises:
            ValueError: If the source sample rate is lower than ``min_sr`` or
                the waveform shape is unsupported.
        """
        import soundfile as sf
        from scipy import signal

        if isinstance(prompt_audio_path, torch.Tensor):
            speech_tensor = prompt_audio_path.detach().cpu().float()
            if speech_tensor.ndim == 1:
                return speech_tensor.unsqueeze(0)
            if speech_tensor.ndim == 2:
                if speech_tensor.shape[0] == 1:
                    return speech_tensor
                if speech_tensor.shape[1] == 1:
                    return speech_tensor.transpose(0, 1)
                return speech_tensor.mean(dim=0, keepdim=True)
            raise ValueError(
                f"Unexpected in-memory prompt audio shape for Wan2.2 S2V TTS mode: {tuple(speech_tensor.shape)}"
            )

        if isinstance(prompt_audio_path, np.ndarray):
            speech = np.asarray(prompt_audio_path, dtype=np.float32)
            if speech.ndim == 1:
                return torch.from_numpy(speech).unsqueeze(0)
            if speech.ndim == 2:
                if speech.shape[0] == 1:
                    return torch.from_numpy(speech.astype(np.float32, copy=False))
                if speech.shape[1] == 1:
                    return torch.from_numpy(speech.T.astype(np.float32, copy=False))
                return torch.from_numpy(speech.mean(axis=1, dtype=np.float32)).unsqueeze(0)
            raise ValueError(
                f"Unexpected numpy prompt audio shape for Wan2.2 S2V TTS mode: {speech.shape}"
            )

        speech, sample_rate = sf.read(prompt_audio_path, dtype="float32", always_2d=False)
        speech = np.asarray(speech, dtype=np.float32)
        if speech.ndim == 2:
            speech = speech.mean(axis=1)
        if speech.ndim != 1:
            raise ValueError(
                f"Unexpected prompt audio shape for Wan2.2 S2V TTS mode: {speech.shape}"
            )
        if sample_rate != target_sr:
            if sample_rate < min_sr:
                raise ValueError(f"wav sample rate {sample_rate} must be at least {min_sr}")
            gcd = math.gcd(sample_rate, target_sr)
            speech = signal.resample_poly(
                speech,
                target_sr // gcd,
                sample_rate // gcd,
            ).astype(np.float32, copy=False)
        return torch.from_numpy(speech).unsqueeze(0)

    def _synthesize_tts_audio(
        self,
        *,
        prompt_audio_path: str,
        prompt_text: str | None,
        text: str,
        output_path: str | None = None,
    ) -> str:
        """Generate a driving-audio file for TTS-driven S2V requests.

        Args:
            prompt_audio_path: Reference speech used for zero-shot or
                cross-lingual synthesis.
            prompt_text: Optional transcript for zero-shot TTS.
            text: Target text to synthesize.
            output_path: Optional destination WAV path.

        Returns:
            The path of the synthesized WAV file.

        Raises:
            RuntimeError: If CosyVoice produces no speech segments.
        """
        self._ensure_tts_loaded()

        import soundfile as sf

        speech_list = []
        prompt_speech_16k = self._load_tts_prompt_audio(prompt_audio_path, 16000)
        if prompt_text is not None:
            iterator = self.pipeline.cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech_16k)
        else:
            iterator = self.pipeline.cosyvoice.inference_cross_lingual(text, prompt_speech_16k)
        for item in iterator:
            speech_list.append(item["tts_speech"])
        if not speech_list:
            raise RuntimeError("CosyVoice returned no synthesized speech for Wan2.2 S2V TTS mode.")
        if output_path is None:
            fd, output_path = tempfile.mkstemp(prefix="wan22_s2v_tts_", suffix=".wav")
            os.close(fd)
        audio_array = torch.cat(speech_list, dim=1).detach().cpu().numpy()
        if audio_array.ndim == 2:
            audio_array = audio_array[0] if audio_array.shape[0] == 1 else np.transpose(audio_array, (1, 0))
        sf.write(output_path, audio_array, self.pipeline.cosyvoice.sample_rate)
        return output_path

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        image: Image.Image,
        audio_path: str | None,
        enable_tts: bool,
        tts_prompt_audio: str | None,
        tts_prompt_text: str | None,
        tts_text: str | None,
        pose_video: str | None,
        height: int,
        width: int,
        num_frames: int | None,
        infer_frames: int | None,
        num_inference_steps: int | None,
        guidance_scale: float | None,
        shift: float | None,
        sample_solver: str,
        seed: int,
        num_repeat: int | None,
        offload_model: bool,
        init_first_frame: bool,
        temp_dir: str,
    ) -> tuple[torch.Tensor, str | None]:
        """Run native Wan2.2 S2V inference for one request.

        Args:
            prompt: Positive prompt describing the target motion.
            negative_prompt: Optional negative prompt.
            image: Reference identity image.
            audio_path: Driving audio path for standard S2V mode.
            enable_tts: Whether to synthesize the driving audio first.
            tts_prompt_audio: Prompt audio path for TTS mode.
            tts_prompt_text: Optional transcript for the prompt audio.
            tts_text: Target text to synthesize in TTS mode.
            pose_video: Optional pose guidance video path.
            height: Output video height.
            width: Output video width.
            num_frames: Requested total frame count.
            infer_frames: Requested Wan-native inference clip length.
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG guidance scale.
            shift: Scheduler shift value.
            sample_solver: Solver name.
            seed: Base random seed. Negative values request random seeding.
            num_repeat: Maximum number of repeated inference windows.
            offload_model: Whether to offload large modules between stages.
            init_first_frame: Whether to seed the first clip with the reference
                frame instead of dropping the initial motion window.
            temp_dir: Request-scoped temporary directory reserved for upstream
                compatibility.

        Returns:
            A tuple containing the decoded video tensor and an optional path to
            synthesized TTS audio.

        Raises:
            ValueError: If required audio inputs are missing.
        """
        del temp_dir
        num_frames, infer_frames = self.resolve_num_frames(num_frames, infer_frames)
        sampling_steps = num_inference_steps or self.default_num_inference_steps
        guide_scale = guidance_scale if guidance_scale is not None else self.default_guidance_scale
        shift = shift if shift is not None else self.default_shift

        generated_audio_path: str | None = None
        if enable_tts:
            if tts_prompt_audio is None:
                raise ValueError("Wan2.2 S2V TTS mode requires `tts_prompt_audio`.")
            if not isinstance(tts_text, str) or not tts_text.strip():
                raise ValueError("Wan2.2 S2V TTS mode requires a non-empty `tts_text`.")
            generated_audio_path = self._synthesize_tts_audio(
                prompt_audio_path=tts_prompt_audio,
                prompt_text=tts_prompt_text,
                text=tts_text,
            )
            audio_path = generated_audio_path
        elif audio_path is None:
            raise ValueError("Wan2.2 S2V requires `audio_path` when TTS mode is disabled.")

        channel = 3
        resize_op = transforms.Resize(min(height, width))
        crop_op = transforms.CenterCrop((height, width))
        tensor_to_image = transforms.ToTensor()

        ref_image = np.array(image.convert("RGB"))
        motion_latents = torch.zeros(
            [1, channel, self.motion_frames, height, width],
            dtype=self.pipeline.param_dtype,
            device=self.device,
        )

        audio_emb, backend_num_repeat = self.pipeline.encode_audio(audio_path, infer_frames=infer_frames)
        if num_repeat is None or num_repeat > backend_num_repeat:
            num_repeat = backend_num_repeat

        lat_motion_frames = (self.motion_frames + 3) // 4
        model_pic = crop_op(resize_op(Image.fromarray(ref_image)))
        ref_pixel_values = tensor_to_image(model_pic).unsqueeze(1).unsqueeze(0) * 2 - 1.0
        ref_pixel_values = ref_pixel_values.to(dtype=self.pipeline.vae.dtype, device=self.pipeline.vae.device)
        ref_latents = torch.stack(self.pipeline.vae.encode(ref_pixel_values))

        videos_last_frames = motion_latents.detach()
        drop_first_motion = self.pipeline.drop_first_motion
        if init_first_frame:
            drop_first_motion = False
            ref_motion_window = ref_pixel_values.expand(-1, -1, motion_latents[:, :, -6:].shape[2], -1, -1)
            motion_latents[:, :, -6:] = ref_motion_window
        motion_latents = torch.stack(self.pipeline.vae.encode(motion_latents))

        cond_sequences = self.pipeline.load_pose_cond(
            pose_video=pose_video,
            num_repeat=num_repeat,
            infer_frames=infer_frames,
            size=(height, width),
        )

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        negative_prompt = negative_prompt or self.pipeline.sample_neg_prompt

        if not self.pipeline.t5_cpu:
            self.pipeline.text_encoder.model.to(self.device)
            context = self.pipeline.text_encoder([prompt], self.device)
            context_null = self.pipeline.text_encoder([negative_prompt], self.device)
            if offload_model:
                self.pipeline.text_encoder.model.cpu()
        else:
            context = self.pipeline.text_encoder([prompt], torch.device("cpu"))
            context_null = self.pipeline.text_encoder([negative_prompt], torch.device("cpu"))
            context = [tensor.to(self.device) for tensor in context]
            context_null = [tensor.to(self.device) for tensor in context_null]

        outputs = []
        with torch.amp.autocast("cuda", dtype=self.pipeline.param_dtype), torch.no_grad():
            for repeat_idx in range(num_repeat):
                seed_generator = torch.Generator(device=self.device)
                seed_generator.manual_seed(seed + repeat_idx)

                lat_target_frames = (infer_frames + 3 + self.motion_frames) // 4 - lat_motion_frames
                target_shape = [lat_target_frames, height // 8, width // 8]
                latents = [
                    torch.randn(
                        16,
                        target_shape[0],
                        target_shape[1],
                        target_shape[2],
                        dtype=self.pipeline.param_dtype,
                        device=self.device,
                        generator=seed_generator,
                    )
                ]
                max_seq_len = int(np.prod(target_shape) // 4)
                sample_scheduler, timesteps = self._build_scheduler(sample_solver, sampling_steps, shift)

                left_idx = repeat_idx * infer_frames
                right_idx = left_idx + infer_frames
                cond_latents = cond_sequences[repeat_idx] if pose_video else cond_sequences[0] * 0
                cond_latents = cond_latents.to(dtype=self.pipeline.param_dtype, device=self.device)
                audio_input = audio_emb[..., left_idx:right_idx]
                input_motion_latents = motion_latents.clone()

                arg_c = {
                    "context": context[0:1],
                    "seq_len": max_seq_len,
                    "cond_states": cond_latents,
                    "motion_latents": input_motion_latents,
                    "ref_latents": ref_latents,
                    "audio_input": audio_input,
                    "motion_frames": [self.motion_frames, lat_motion_frames],
                    "drop_motion_frames": drop_first_motion and repeat_idx == 0,
                }
                if guide_scale > 1.0:
                    arg_null = {
                        **arg_c,
                        "context": context_null[0:1],
                        "audio_input": 0.0 * audio_input,
                    }
                else:
                    arg_null = None

                if offload_model or self.pipeline.init_on_cpu:
                    self.pipeline.noise_model.to(self.device)
                    torch.cuda.empty_cache()

                for timestep in timesteps:
                    timestep_tensor = torch.stack([timestep]).to(self.device)
                    noise_pred_cond = self.pipeline.noise_model(latents[0:1], t=timestep_tensor, **arg_c)

                    if arg_null is not None:
                        noise_pred_uncond = self.pipeline.noise_model(latents[0:1], t=timestep_tensor, **arg_null)
                        noise_pred = [
                            uncond + guide_scale * (cond - uncond)
                            for cond, uncond in zip(noise_pred_cond, noise_pred_uncond)
                        ]
                    else:
                        noise_pred = noise_pred_cond

                    scheduler_output = sample_scheduler.step(
                        noise_pred[0].unsqueeze(0),
                        timestep,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=seed_generator,
                    )[0]
                    latents[0] = scheduler_output.squeeze(0)

                if offload_model:
                    self.pipeline.noise_model.cpu()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                latents = torch.stack(latents)
                if not (drop_first_motion and repeat_idx == 0):
                    decode_latents = torch.cat([motion_latents, latents], dim=2)
                else:
                    decode_latents = torch.cat([ref_latents, latents], dim=2)

                clip = torch.stack(self.pipeline.vae.decode(decode_latents))
                clip = clip[:, :, -infer_frames:]
                if drop_first_motion and repeat_idx == 0:
                    clip = clip[:, :, 3:]

                overlap_frames_num = min(self.motion_frames, clip.shape[2])
                videos_last_frames = torch.cat(
                    [videos_last_frames[:, :, overlap_frames_num:], clip[:, :, -overlap_frames_num:]],
                    dim=2,
                )
                videos_last_frames = videos_last_frames.to(
                    dtype=motion_latents.dtype,
                    device=motion_latents.device,
                )
                motion_latents = torch.stack(self.pipeline.vae.encode(videos_last_frames))
                outputs.append(clip.cpu() if offload_model else clip)

        videos = torch.cat(outputs, dim=2)
        del latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()

        return videos[0], generated_audio_path
