# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audio and geometry helpers for Wan2.2 S2V request normalization."""

from __future__ import annotations

import logging
import math
import os
import wave
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image

logger = logging.getLogger(__name__)

DEFAULT_S2V_MAX_AREA = 1024 * 704


def get_size_less_than_area(
    height: int,
    width: int,
    target_area: int = DEFAULT_S2V_MAX_AREA,
    divisor: int = 64,
) -> tuple[int, int]:
    """Match Wan2.2 S2V's size search so height/width stay upstream-compatible."""
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got {(height, width)}")

    if height * width <= target_area:
        max_upper_area = target_area
        min_scale = 0.1
        max_scale = 1.0
    else:
        max_upper_area = target_area
        d = divisor - 1
        b = d * (height + width)
        a = height * width
        c = d**2 - max_upper_area

        min_scale = (-b + math.sqrt(b**2 - 2 * a * c)) / (2 * a)
        max_scale = math.sqrt(max_upper_area / (height * width))

    for i in range(100):
        scale = max_scale - (max_scale - min_scale) * i / 100
        new_height = int(height * scale)
        new_width = int(width * scale)

        pad_height = (divisor - new_height % divisor) % divisor
        pad_width = (divisor - new_width % divisor) % divisor
        padded_height = new_height + pad_height
        padded_width = new_width + pad_width

        if padded_height * padded_width <= max_upper_area:
            return padded_height, padded_width

    aspect_ratio = width / height
    target_width = int((target_area * aspect_ratio) ** 0.5 // divisor * divisor)
    target_height = int((target_area / aspect_ratio) ** 0.5 // divisor * divisor)
    if target_width >= width or target_height >= height:
        target_width = max(divisor, int(width // divisor * divisor))
        target_height = max(divisor, int(height // divisor * divisor))
    return target_height, target_width


def choose_s2v_dimensions(
    image: PIL.Image.Image,
    height: int | None,
    width: int | None,
    *,
    max_area: int = DEFAULT_S2V_MAX_AREA,
) -> tuple[int, int]:
    """Resolve Wan2.2 S2V output dimensions for a request.

    Args:
        image: Normalized reference image.
        height: Optional requested height.
        width: Optional requested width.
        max_area: Maximum allowed pixel area when dimensions need to be
            derived from the input image.

    Returns:
        A pair of 64-aligned ``(height, width)`` dimensions accepted by the
        native Wan runtime.
    """
    if height is not None and width is not None:
        return max(64, height // 64 * 64), max(64, width // 64 * 64)
    return get_size_less_than_area(image.height, image.width, target_area=max_area)


def normalize_image_input(image_input: Any) -> PIL.Image.Image:
    """Convert a user-provided image payload into a single RGB image.

    Args:
        image_input: Image payload supplied by the request. Supported types are
            ``PIL.Image.Image``, filesystem paths, or one-element lists of those
            values.

    Returns:
        A normalized RGB ``PIL.Image.Image``.

    Raises:
        FileNotFoundError: If a referenced image path does not exist.
        TypeError: If the payload type is unsupported.
        ValueError: If an empty image list is provided.
    """
    if isinstance(image_input, list):
        if not image_input:
            raise ValueError("Received an empty image list for Wan2.2 S2V input.")
        if len(image_input) > 1:
            logger.warning("Wan2.2 S2V only supports one reference image. Using the first entry.")
        image_input = image_input[0]

    if isinstance(image_input, PIL.Image.Image):
        return image_input.convert("RGB")

    if isinstance(image_input, (str, os.PathLike)):
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
        with PIL.Image.open(image_path) as image:
            return image.convert("RGB")

    raise TypeError(
        "Wan2.2 S2V expects `multi_modal_data[\"image\"]` to be a file path or PIL.Image.Image."
    )


def normalize_pose_video_input(pose_video: Any) -> str | None:
    """Normalize an optional pose-video reference into a filesystem path.

    Args:
        pose_video: Optional pose-video payload. Only filesystem paths or
            one-element lists of paths are supported.

    Returns:
        The normalized path string or ``None`` when pose guidance is absent.

    Raises:
        FileNotFoundError: If the referenced path does not exist.
        TypeError: If the payload type is unsupported.
    """
    if pose_video is None:
        return None
    if isinstance(pose_video, list):
        if not pose_video:
            return None
        if len(pose_video) > 1:
            logger.warning("Wan2.2 S2V only supports one pose video. Using the first entry.")
        pose_video = pose_video[0]

    if isinstance(pose_video, (str, os.PathLike)):
        pose_path = Path(pose_video)
        if not pose_path.exists():
            raise FileNotFoundError(f"Pose video path does not exist: {pose_path}")
        return str(pose_path)

    raise TypeError("Wan2.2 S2V only supports `pose_video` as a filesystem path.")


def _coerce_audio_array(audio_array: np.ndarray) -> np.ndarray:
    """Convert an audio array to mono float32 samples.

    Args:
        audio_array: Input waveform in either mono or common stereo layouts.

    Returns:
        A one-dimensional float32 waveform suitable for WAV serialization.

    Raises:
        ValueError: If the audio shape is empty or unsupported.
    """
    array = np.asarray(audio_array)
    if array.ndim == 0:
        raise ValueError("Audio array must contain at least one sample.")
    if array.ndim == 1:
        return array.astype(np.float32, copy=False)
    if array.ndim == 2:
        if array.shape[0] in (1, 2) and array.shape[1] not in (1, 2):
            mono = array.mean(axis=0)
        elif array.shape[1] in (1, 2) and array.shape[0] not in (1, 2):
            mono = array.mean(axis=1)
        elif array.shape[0] in (1, 2) and array.shape[1] in (1, 2):
            logger.warning(
                "Ambiguous 2D audio shape %s for Wan2.2 S2V; assuming [channels, samples].",
                array.shape,
            )
            mono = array.mean(axis=0)
        elif array.shape[0] < array.shape[1]:
            logger.warning(
                "Uncommon 2D audio shape %s for Wan2.2 S2V; assuming [channels, samples].",
                array.shape,
            )
            mono = array.mean(axis=0)
        else:
            logger.warning(
                "Uncommon 2D audio shape %s for Wan2.2 S2V; assuming [samples, channels].",
                array.shape,
            )
            mono = array.mean(axis=1)
        return mono.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported audio array shape for Wan2.2 S2V: {array.shape}")


def write_wav(audio_path: str | os.PathLike[str], audio_array: np.ndarray, sample_rate: int) -> str:
    """Serialize an in-memory waveform to a mono 16-bit PCM WAV file.

    Args:
        audio_path: Destination path.
        audio_array: Input waveform array.
        sample_rate: Sampling rate in Hz.

    Returns:
        The destination path as a string for convenient chaining.

    Raises:
        ValueError: If ``sample_rate`` is not positive.
    """
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    samples = np.clip(_coerce_audio_array(audio_array), -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2")

    audio_path = str(audio_path)
    with wave.open(audio_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return audio_path


def normalize_audio_request_input(
    audio_input: Any,
) -> str | tuple[np.ndarray, int] | np.ndarray:
    """Validate and canonicalize request audio without materializing a temp WAV.

    Args:
        audio_input: Audio payload supplied by the request. Supported forms are
            filesystem paths, bare numpy arrays, ``(array, sample_rate)`` tuples,
            or one-element lists of those values.

    Returns:
        A normalized path, mono waveform array, or ``(waveform, sample_rate)``
        tuple that can later be serialized by :func:`normalize_audio_input`.

    Raises:
        FileNotFoundError: If a referenced audio path does not exist.
        TypeError: If the payload type is unsupported.
        ValueError: If the payload is empty or invalid.
    """
    if isinstance(audio_input, list):
        if not audio_input:
            raise ValueError("Received an empty audio list for Wan2.2 S2V input.")
        if len(audio_input) > 1:
            logger.warning("Wan2.2 S2V only supports one driving audio input. Using the first entry.")
        audio_input = audio_input[0]

    if isinstance(audio_input, (str, os.PathLike)):
        audio_path = Path(audio_input)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio path does not exist: {audio_path}")
        return str(audio_path)

    if isinstance(audio_input, tuple) and len(audio_input) == 2:
        audio_array, sample_rate = audio_input
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        return _coerce_audio_array(np.asarray(audio_array)), sample_rate

    if isinstance(audio_input, np.ndarray):
        return _coerce_audio_array(audio_input)

    raise TypeError(
        "Wan2.2 S2V expects `multi_modal_data[\"audio\"]` to be a file path, a numpy array, or "
        "a `(np.ndarray, sample_rate)` tuple."
    )


def normalize_audio_input(audio_input: Any, temp_dir: str | os.PathLike[str]) -> str:
    """Normalize a request audio payload into a local WAV path.

    Args:
        audio_input: Audio payload supplied by the request. Supported forms are
            filesystem paths, bare numpy arrays, ``(array, sample_rate)`` tuples,
            or one-element lists of those values.
        temp_dir: Temporary directory used when the input must be materialized
            as a WAV file.

    Returns:
        A filesystem path pointing to a readable WAV file.

    Raises:
        FileNotFoundError: If a referenced audio path does not exist.
        TypeError: If the payload type is unsupported.
        ValueError: If an empty audio list is provided.
    """
    normalized_audio = normalize_audio_request_input(audio_input)

    if isinstance(normalized_audio, str):
        return normalized_audio

    if isinstance(normalized_audio, tuple):
        audio_array, sample_rate = normalized_audio
        audio_path = Path(temp_dir) / "wan22_s2v_input.wav"
        return write_wav(audio_path, audio_array, sample_rate)

    if isinstance(normalized_audio, np.ndarray):
        audio_path = Path(temp_dir) / "wan22_s2v_input.wav"
        logger.warning(
            "Received a bare numpy audio array for Wan2.2 S2V. Assuming 16kHz sample rate."
        )
        return write_wav(audio_path, normalized_audio, 16000)

    raise TypeError(f"Unsupported normalized audio input type: {type(normalized_audio)!r}")
