# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online-serving smoke tests for the Wan2.2 S2V API expansion path."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
import requests
import torch

from tests.conftest import OmniServer, OmniServerParams, assert_video_valid

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

MODEL_DIR_ENV = "VLLM_TEST_WAN22_S2V_MODEL"
RUN_ENV = "RUN_WAN22_S2V_E2E"
ASSET_ROOT = Path(__file__).resolve().parents[3] / "upstream_refs" / "Wan2.2" / "examples"
IMAGE_PATH = ASSET_ROOT / "i2v_input.JPG"
AUDIO_PATH = ASSET_ROOT / "talk.wav"
TTS_PROMPT_AUDIO_PATH = ASSET_ROOT / "zero_shot_prompt.wav"
MODEL_DIR = (
    Path(model_dir).expanduser().resolve()
    if (model_dir := os.environ.get(MODEL_DIR_ENV))
    else None
)
VIDEO_POLL_INTERVAL_S = 2.0
VIDEO_TIMEOUT_S = 1800.0


def _skip_reason() -> str:
    """Return the reason these opt-in smoke tests should be skipped."""
    if os.environ.get(RUN_ENV) != "1":
        return f"Set {RUN_ENV}=1 to run Wan2.2 S2V online serving smoke tests."
    if not torch.cuda.is_available():
        return "Wan2.2 S2V online serving smoke tests require CUDA."
    if MODEL_DIR is None:
        return f"Set {MODEL_DIR_ENV} to the Wan2.2 S2V model directory."
    if not MODEL_DIR.exists():
        return f"Wan2.2 S2V model directory not found: {MODEL_DIR}"
    missing_assets = [path for path in (IMAGE_PATH, AUDIO_PATH, TTS_PROMPT_AUDIO_PATH) if not path.exists()]
    if missing_assets:
        return f"Wan2.2 example assets are missing: {', '.join(str(path) for path in missing_assets)}"
    return ""


SKIP_REASON = _skip_reason()
RUN_CASE = SKIP_REASON == ""
TEST_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=str(MODEL_DIR),
            server_args=["--enable-cpu-offload", "--enforce-eager", "--disable-log-stats"],
            init_timeout=1800,
        ),
        id="wan22_s2v_sync",
        marks=pytest.mark.skipif(not RUN_CASE, reason=SKIP_REASON or "Wan2.2 S2V smoke tests disabled."),
    )
]


def _video_sync_url(server: OmniServer) -> str:
    """Build the synchronous video endpoint URL for the test server."""
    return f"http://{server.host}:{server.port}/v1/videos/sync"


def _video_async_url(server: OmniServer, suffix: str = "") -> str:
    """Build an asynchronous video endpoint URL for the test server."""
    return f"http://{server.host}:{server.port}/v1/videos{suffix}"


def _base_form() -> dict[str, str]:
    """Return the shared multipart form fields for Wan2.2 S2V requests."""
    return {
        "prompt": "A person is talking naturally to the camera.",
        "width": "832",
        "height": "448",
        "num_frames": "17",
        "num_inference_steps": "2",
        "guidance_scale": "4.5",
        "seed": "42",
        "extra_params": json.dumps(
            {
                "infer_frames": 16,
                "num_repeat": 1,
                "shift": 3.0,
                "solver_name": "unipc",
                "offload_model": True,
            }
        ),
    }


def _wait_for_video_status(server: OmniServer, video_id: str, expected_status: str) -> dict[str, Any]:
    """Poll an async video job until it reaches the expected terminal state."""
    deadline = time.time() + VIDEO_TIMEOUT_S
    last_payload: dict[str, Any] | None = None

    while time.time() < deadline:
        response = requests.get(_video_async_url(server, f"/{video_id}"), timeout=VIDEO_TIMEOUT_S)
        assert response.status_code == 200, response.text
        last_payload = response.json()
        status = last_payload["status"]
        if status == expected_status:
            return last_payload
        if status == "failed":
            raise AssertionError(f"Video job {video_id} failed unexpectedly: {last_payload}")
        time.sleep(VIDEO_POLL_INTERVAL_S)

    raise AssertionError(
        f"Timed out waiting for video job {video_id} to reach status={expected_status}. Last payload: {last_payload}"
    )


def _delete_video_job(server: OmniServer, video_id: str) -> requests.Response:
    """Delete an async video job through the API."""
    return requests.delete(_video_async_url(server, f"/{video_id}"), timeout=VIDEO_TIMEOUT_S)


def _delete_video_job_with_retry(server: OmniServer, video_id: str) -> requests.Response:
    """Retry deletion until the async job leaves the in-progress state."""
    deadline = time.time() + 30.0
    last_response: requests.Response | None = None

    while time.time() < deadline:
        response = _delete_video_job(server, video_id)
        last_response = response
        if response.status_code != 409:
            return response
        time.sleep(1.0)

    raise AssertionError(
        f"Timed out waiting to delete video job {video_id}. "
        f"Last response: {None if last_response is None else last_response.text}"
    )


def _best_effort_delete(server: OmniServer, video_id: str) -> None:
    """Attempt cleanup without masking the primary test result."""
    try:
        response = _delete_video_job_with_retry(server, video_id)
        if response.status_code not in (200, 404):
            print(f"Cleanup delete for {video_id} returned {response.status_code}: {response.text}")
    except Exception as exc:
        print(f"Cleanup delete for {video_id} failed: {exc}")


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
def test_wan22_s2v_sync_audio_driven(omni_server: OmniServer) -> None:
    """Verify sync audio-driven S2V generation through the video API."""
    with IMAGE_PATH.open("rb") as image_file, AUDIO_PATH.open("rb") as audio_file:
        response = requests.post(
            _video_sync_url(omni_server),
            data=_base_form(),
            files={
                "input_reference": (IMAGE_PATH.name, image_file, "image/jpeg"),
                "input_audio_reference": (AUDIO_PATH.name, audio_file, "audio/wav"),
            },
            timeout=1800,
        )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("video/mp4")
    assert_video_valid(response.content, width=832, height=448, fps=16)


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
def test_wan22_s2v_sync_tts_driven(omni_server: OmniServer) -> None:
    """Verify sync TTS-driven S2V generation through the video API."""
    form_data = _base_form()
    form_data.update(
        {
            "enable_tts": "true",
            "tts_prompt_text": "希望你以后能够做的比我还好呦。",
            "tts_text": "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。",
        }
    )

    with IMAGE_PATH.open("rb") as image_file, TTS_PROMPT_AUDIO_PATH.open("rb") as prompt_audio_file:
        response = requests.post(
            _video_sync_url(omni_server),
            data=form_data,
            files={
                "input_reference": (IMAGE_PATH.name, image_file, "image/jpeg"),
                "input_tts_prompt_audio": (TTS_PROMPT_AUDIO_PATH.name, prompt_audio_file, "audio/wav"),
            },
            timeout=1800,
        )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("video/mp4")
    assert_video_valid(response.content, width=832, height=448, fps=16)


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
def test_wan22_s2v_async_audio_driven(omni_server: OmniServer, tmp_path: Path) -> None:
    """Verify async audio-driven S2V generation and artifact retrieval."""
    video_id: str | None = None
    try:
        with IMAGE_PATH.open("rb") as image_file, AUDIO_PATH.open("rb") as audio_file:
            response = requests.post(
                _video_async_url(omni_server),
                data=_base_form(),
                files={
                    "input_reference": (IMAGE_PATH.name, image_file, "image/jpeg"),
                    "input_audio_reference": (AUDIO_PATH.name, audio_file, "audio/wav"),
                },
                timeout=VIDEO_TIMEOUT_S,
            )

        assert response.status_code == 200, response.text
        created = response.json()
        video_id = created["id"]
        assert created["object"] == "video"
        assert created["status"] == "queued"

        completed = _wait_for_video_status(omni_server, video_id, "completed")
        assert completed["file_name"] is not None
        assert completed["progress"] == 100

        retrieve_response = requests.get(_video_async_url(omni_server, f"/{video_id}"), timeout=VIDEO_TIMEOUT_S)
        assert retrieve_response.status_code == 200, retrieve_response.text
        assert retrieve_response.json()["status"] == "completed"

        list_response = requests.get(_video_async_url(omni_server), timeout=VIDEO_TIMEOUT_S)
        assert list_response.status_code == 200, list_response.text
        assert any(item["id"] == video_id for item in list_response.json()["data"])

        download_response = requests.get(_video_async_url(omni_server, f"/{video_id}/content"), timeout=VIDEO_TIMEOUT_S)
        assert download_response.status_code == 200, download_response.text
        assert download_response.headers["content-type"].startswith("video/mp4")
        assert_video_valid(download_response.content, width=832, height=448, fps=16)

        output_path = tmp_path / completed["file_name"]
        output_path.write_bytes(download_response.content)
        assert output_path.stat().st_size == len(download_response.content)
    finally:
        if video_id is not None:
            _best_effort_delete(omni_server, video_id)


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
def test_wan22_s2v_async_tts_driven(omni_server: OmniServer, tmp_path: Path) -> None:
    """Verify async TTS-driven S2V generation and artifact retrieval."""
    video_id: str | None = None
    form_data = _base_form()
    form_data.update(
        {
            "enable_tts": "true",
            "tts_prompt_text": "希望你以后能够做的比我还好呦。",
            "tts_text": "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。",
        }
    )

    try:
        with IMAGE_PATH.open("rb") as image_file, TTS_PROMPT_AUDIO_PATH.open("rb") as prompt_audio_file:
            response = requests.post(
                _video_async_url(omni_server),
                data=form_data,
                files={
                    "input_reference": (IMAGE_PATH.name, image_file, "image/jpeg"),
                    "input_tts_prompt_audio": (TTS_PROMPT_AUDIO_PATH.name, prompt_audio_file, "audio/wav"),
                },
                timeout=VIDEO_TIMEOUT_S,
            )

        assert response.status_code == 200, response.text
        created = response.json()
        video_id = created["id"]
        assert created["object"] == "video"
        assert created["status"] == "queued"

        completed = _wait_for_video_status(omni_server, video_id, "completed")
        assert completed["file_name"] is not None
        assert completed["progress"] == 100

        download_response = requests.get(_video_async_url(omni_server, f"/{video_id}/content"), timeout=VIDEO_TIMEOUT_S)
        assert download_response.status_code == 200, download_response.text
        assert download_response.headers["content-type"].startswith("video/mp4")
        assert_video_valid(download_response.content, width=832, height=448, fps=16)

        output_path = tmp_path / completed["file_name"]
        output_path.write_bytes(download_response.content)
        assert output_path.stat().st_size == len(download_response.content)
    finally:
        if video_id is not None:
            _best_effort_delete(omni_server, video_id)
