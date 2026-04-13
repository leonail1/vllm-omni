# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline Wan2.2 S2V example that exercises the vLLM-Omni integration."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the offline Wan2.2 S2V demo.

    Returns:
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(description="Offline Wan2.2 S2V inference via vLLM-Omni.")
    parser.add_argument("--model", required=True, help="Local Wan2.2-S2V-14B checkpoint root.")
    parser.add_argument("--prompt", required=True, help="Driving text prompt.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt.")
    parser.add_argument("--image", required=True, help="Reference image path.")
    parser.add_argument("--audio", default=None, help="Driving audio path. Optional when --enable-tts is set.")
    parser.add_argument("--pose-video", default=None, help="Optional pose video path.")
    parser.add_argument("--enable-tts", action="store_true", help="Synthesize the driving audio with CosyVoice first.")
    parser.add_argument("--tts-prompt-audio", default=None, help="Reference speech for CosyVoice zero-shot / cross-lingual TTS.")
    parser.add_argument("--tts-prompt-text", default=None, help="Optional transcript for the TTS prompt audio.")
    parser.add_argument("--tts-text", default=None, help="Target text that CosyVoice should synthesize before S2V.")
    parser.add_argument("--height", type=int, default=448, help="Output height (64-aligned).")
    parser.add_argument("--width", type=int, default=832, help="Output width.")
    parser.add_argument("--num-frames", type=int, default=17, help="Requested output frame count (4n+1).")
    parser.add_argument("--infer-frames", type=int, default=16, help="Wan native per-clip frame count (4n).")
    parser.add_argument("--num-repeat", type=int, default=1, help="Maximum number of clips to generate.")
    parser.add_argument("--num-inference-steps", type=int, default=8, help="Sampling steps.")
    parser.add_argument("--guidance-scale", type=float, default=4.5, help="CFG guidance scale.")
    parser.add_argument("--shift", type=float, default=3.0, help="Scheduler shift.")
    parser.add_argument("--solver-name", default="unipc", choices=["unipc", "dpm++"], help="Sampling solver.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", default="wan22_s2v.mp4", help="Output mp4 path.")
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable native Wan CPU offload inside the vLLM-Omni wrapper.",
    )
    parser.add_argument(
        "--merge-audio",
        action="store_true",
        help="Mux the driving or synthesized audio into the output video with ffmpeg when available.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph capture in Omni to reduce first-run initialization overhead.",
    )
    return parser.parse_args()


def _save_video(video, output_path: str, fps: int) -> None:
    """Serialize a generated video payload to an MP4 file.

    Args:
        video: Video payload returned by ``Omni.generate``.
        output_path: Destination MP4 path.
        fps: Output frames per second.

    Raises:
        ValueError: If the payload shape cannot be interpreted as a video.
    """
    if hasattr(video, "detach"):
        video = video.detach().cpu().numpy()
    video = np.asarray(video)
    if video.ndim == 5 and video.shape[0] == 1:
        video = video[0]
    if video.ndim == 5 and video.shape[1] in (3, 4):
        video = np.transpose(video[0], (1, 2, 3, 0))
    elif video.ndim == 4 and video.shape[0] in (3, 4):
        video = np.transpose(video, (1, 2, 3, 0))

    if video.ndim != 4:
        raise ValueError(f"Unexpected video shape for export: {video.shape}")

    if np.issubdtype(video.dtype, np.floating):
        if video.min() < 0.0 or video.max() > 1.0:
            video = np.clip(video, -1.0, 1.0) * 0.5 + 0.5
        video = (np.clip(video, 0.0, 1.0) * 255).round().astype(np.uint8)

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    try:
        for frame in video:
            writer.append_data(frame)
    finally:
        writer.close()


def _merge_audio(video_path: str, audio_path: str) -> None:
    """Mux an audio track into an existing MP4 with ffmpeg.

    Args:
        video_path: Path to the generated MP4 file.
        audio_path: Path to the WAV/audio file to merge.

    Raises:
        RuntimeError: If ``ffmpeg`` is unavailable.
        CalledProcessError: If the ffmpeg invocation fails.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for --merge-audio but was not found in PATH.")

    temp_path = str(Path(video_path).with_suffix(".mux.mp4"))
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        temp_path,
    ]
    subprocess.run(cmd, check=True)
    Path(temp_path).replace(video_path)


def _validate_args(args: argparse.Namespace) -> None:
    """Validate mutually dependent CLI arguments.

    Args:
        args: Parsed CLI namespace.

    Raises:
        ValueError: If required audio/TTS arguments are missing.
    """
    if not args.enable_tts and args.audio is None:
        raise ValueError("Pass --audio when TTS mode is disabled.")
    if args.enable_tts and (args.tts_prompt_audio is None or args.tts_text is None):
        raise ValueError("TTS mode requires both --tts-prompt-audio and --tts-text.")


def main() -> None:
    """Run the offline Wan2.2 S2V demo end to end."""
    args = parse_args()
    _validate_args(args)

    omni = Omni(
        model=args.model,
        enable_cpu_offload=args.enable_cpu_offload,
        enforce_eager=args.enforce_eager,
    )

    multi_modal_data = {
        "image": args.image,
        "pose_video": args.pose_video,
    }
    if args.audio is not None:
        multi_modal_data["audio"] = args.audio

    prompt = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "multi_modal_data": multi_modal_data,
    }

    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        extra_args={
            "infer_frames": args.infer_frames,
            "num_repeat": args.num_repeat,
            "shift": args.shift,
            "solver_name": args.solver_name,
            "offload_model": args.enable_cpu_offload,
            "enable_tts": args.enable_tts,
            "tts_prompt_audio": args.tts_prompt_audio,
            "tts_prompt_text": args.tts_prompt_text,
            "tts_text": args.tts_text,
        },
    )

    start = time.perf_counter()
    result = omni.generate(prompt, sampling_params)
    elapsed = time.perf_counter() - start

    if isinstance(result, list):
        result = result[0] if result else None
    if result is None or not isinstance(result, OmniRequestOutput):
        raise RuntimeError("Wan2.2 S2V returned no valid OmniRequestOutput.")
    if not result.images:
        raise RuntimeError("Wan2.2 S2V returned an empty video payload.")

    video = result.images[0]
    fps = int(result.multimodal_output.get("fps", 16))
    generated_audio_path = result.custom_output.get("audio_path")
    cleanup_generated_audio = bool(result.custom_output.get("cleanup_audio_path"))

    generated_audio_removed = False

    _save_video(video, args.output, fps)
    if args.merge_audio:
        audio_to_merge = generated_audio_path or args.audio
        if audio_to_merge is not None:
            _merge_audio(args.output, audio_to_merge)
            if cleanup_generated_audio and generated_audio_path is not None and audio_to_merge == generated_audio_path:
                Path(generated_audio_path).unlink(missing_ok=True)
                generated_audio_removed = True
        else:
            print("Skipped audio mux because no mergeable audio path was produced.")

    print(f"Saved video to {args.output}")
    if generated_audio_path is not None:
        if generated_audio_removed:
            print("Synthesized audio was merged into the output video and the temporary file was removed.")
        else:
            print(f"Synthesized audio: {generated_audio_path}")
    print(f"Video shape: {getattr(video, 'shape', None)}")
    print(f"FPS: {fps}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
