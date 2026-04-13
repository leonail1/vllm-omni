# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Wan2.2 S2V request pre-processing helpers."""

from __future__ import annotations

import numpy as np
import PIL.Image
import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_s2v import get_wan22_s2v_pre_process_func
from vllm_omni.diffusion.models.wan2_2.wan2_2_s2v_audio import _coerce_audio_array, normalize_audio_input
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def test_wan22_s2v_preprocess_normalizes_image_audio_and_request_defaults():
    """Verify that preprocessing fills in Wan-native defaults and normalizes inputs."""
    pre_process = get_wan22_s2v_pre_process_func(OmniDiffusionConfig(model="/tmp/fake-s2v"))
    request = OmniDiffusionRequest(
        prompts=[
            {
                "prompt": "test prompt",
                "multi_modal_data": {
                    "image": PIL.Image.new("RGB", (640, 360), color=(120, 80, 40)),
                    "audio": (np.zeros((16000,), dtype=np.float32), 16000),
                },
            }
        ],
        sampling_params=OmniDiffusionSamplingParams(),
    )

    processed = pre_process(request)
    prompt = processed.prompts[0]
    image = prompt["multi_modal_data"]["image"]

    assert isinstance(image, PIL.Image.Image)
    assert processed.sampling_params.num_frames == 81
    assert processed.sampling_params.fps == 16
    assert processed.sampling_params.frame_rate == 16.0
    assert processed.sampling_params.height % 64 == 0
    assert processed.sampling_params.width % 64 == 0
    assert image.size == (640, 360)
    assert prompt["multi_modal_data"]["audio"][1] == 16000
    assert prompt["multi_modal_data"]["audio"][0].dtype == np.float32
    assert prompt["additional_information"]["preprocessed_image"] is image


def test_wan22_s2v_normalize_audio_input_accepts_bare_numpy_array(tmp_path):
    """Verify that bare numpy audio arrays are materialized as WAV files."""
    audio = np.zeros((16000,), dtype=np.float32)

    audio_path = normalize_audio_input(audio, tmp_path)

    assert audio_path.endswith(".wav")
    assert tmp_path.joinpath("wan22_s2v_input.wav").exists()


def test_wan22_s2v_preprocess_accepts_tts_mode_without_audio():
    """Verify that TTS-driven requests do not require driving audio up front."""
    pre_process = get_wan22_s2v_pre_process_func(OmniDiffusionConfig(model="/tmp/fake-s2v"))
    request = OmniDiffusionRequest(
        prompts=[
            {
                "prompt": "test prompt",
                "multi_modal_data": {
                    "image": PIL.Image.new("RGB", (640, 360), color=(120, 80, 40)),
                },
            }
        ],
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={
                "enable_tts": True,
                "tts_prompt_audio": "/tmp/fake_prompt.wav",
                "tts_text": "test speech",
            }
        ),
    )

    processed = pre_process(request)
    prompt = processed.prompts[0]

    assert "audio" in prompt["multi_modal_data"]
    assert prompt["multi_modal_data"]["audio"] is None
    assert prompt["additional_information"]["preprocessed_image"].size == (640, 360)


def test_wan22_s2v_preprocess_rejects_missing_audio_path_early():
    """Verify that preprocess validates audio paths before backend execution."""
    pre_process = get_wan22_s2v_pre_process_func(OmniDiffusionConfig(model="/tmp/fake-s2v"))
    request = OmniDiffusionRequest(
        prompts=[
            {
                "prompt": "test prompt",
                "multi_modal_data": {
                    "image": PIL.Image.new("RGB", (640, 360), color=(120, 80, 40)),
                    "audio": "/tmp/does-not-exist.wav",
                },
            }
        ],
        sampling_params=OmniDiffusionSamplingParams(),
    )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        pre_process(request)


def test_wan22_s2v_preprocess_rejects_tts_mode_without_prompt_audio():
    """Verify that TTS mode requires a prompt-audio reference."""
    pre_process = get_wan22_s2v_pre_process_func(OmniDiffusionConfig(model="/tmp/fake-s2v"))
    request = OmniDiffusionRequest(
        prompts=[
            {
                "prompt": "test prompt",
                "multi_modal_data": {
                    "image": PIL.Image.new("RGB", (640, 360), color=(120, 80, 40)),
                },
            }
        ],
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={
                "enable_tts": True,
                "tts_text": "test speech",
            }
        ),
    )

    with pytest.raises(ValueError, match="tts_prompt_audio"):
        pre_process(request)


def test_wan22_s2v_preprocess_rejects_tts_mode_with_empty_tts_text():
    """Verify that TTS mode rejects blank synthesized text."""
    pre_process = get_wan22_s2v_pre_process_func(OmniDiffusionConfig(model="/tmp/fake-s2v"))
    request = OmniDiffusionRequest(
        prompts=[
            {
                "prompt": "test prompt",
                "multi_modal_data": {
                    "image": PIL.Image.new("RGB", (640, 360), color=(120, 80, 40)),
                },
            }
        ],
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={
                "enable_tts": True,
                "tts_prompt_audio": "/tmp/fake_prompt.wav",
                "tts_text": "   ",
            }
        ),
    )

    with pytest.raises(ValueError, match="tts_text"):
        pre_process(request)


def test_wan22_s2v_coerce_audio_array_supports_common_stereo_layouts():
    """Verify that common stereo layouts collapse to the same mono waveform."""
    channels_first = np.asarray(
        [
            [0.0, 0.5, -0.5],
            [1.0, -0.5, 0.5],
        ],
        dtype=np.float32,
    )
    samples_first = channels_first.T

    np.testing.assert_allclose(
        _coerce_audio_array(channels_first),
        np.asarray([0.5, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        _coerce_audio_array(samples_first),
        np.asarray([0.5, 0.0, 0.0], dtype=np.float32),
    )
