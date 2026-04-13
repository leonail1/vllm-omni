# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional local smoke test for offline Wan2.2 S2V inference."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

MODEL_DIR_ENV = "VLLM_TEST_WAN22_S2V_MODEL"
RUN_ENV = "RUN_WAN22_S2V_E2E"
ASSET_ROOT = Path(__file__).resolve().parents[3] / "upstream_refs" / "Wan2.2" / "examples"


@pytest.mark.core_model
@pytest.mark.diffusion
def test_wan22_s2v_local_smoke() -> None:
    """Run a minimal offline Wan2.2 S2V request against local assets."""
    if os.environ.get(RUN_ENV) != "1":
        pytest.skip(f"Set {RUN_ENV}=1 to run the local Wan2.2 S2V smoke test.")
    if not torch.cuda.is_available():
        pytest.skip("Wan2.2 S2V smoke test requires CUDA.")

    model_dir_value = os.environ.get(MODEL_DIR_ENV)
    if not model_dir_value:
        pytest.skip(f"Set {MODEL_DIR_ENV} to the Wan2.2 S2V model directory.")
    model_dir = Path(model_dir_value).expanduser().resolve()
    if not model_dir.exists():
        pytest.skip(f"Wan2.2 S2V model directory not found: {model_dir}")
    image_path = ASSET_ROOT / "i2v_input.JPG"
    audio_path = ASSET_ROOT / "talk.wav"
    if not image_path.exists() or not audio_path.exists():
        pytest.skip("Wan2.2 example assets are missing.")

    omni = Omni(
        model=str(model_dir),
        enable_cpu_offload=True,
        enforce_eager=True,
    )

    result = omni.generate(
        {
            "prompt": "A person is talking naturally to the camera.",
            "multi_modal_data": {
                "image": str(image_path),
                "audio": str(audio_path),
            },
        },
        OmniDiffusionSamplingParams(
            height=448,
            width=832,
            num_frames=17,
            num_inference_steps=2,
            guidance_scale=4.5,
            seed=42,
            extra_args={
                "infer_frames": 16,
                "num_repeat": 1,
                "shift": 3.0,
                "solver_name": "unipc",
                "offload_model": True,
            },
        ),
    )

    if isinstance(result, list):
        result = result[0] if result else None

    assert isinstance(result, OmniRequestOutput)
    assert result.images
    video = np.asarray(result.images[0])
    if video.ndim == 5 and video.shape[0] == 1:
        video = video[0]
    assert video.ndim == 4
    assert video.shape[0] > 0
    assert video.shape[1] == 448
    assert video.shape[2] == 832
    assert int(result.multimodal_output.get("fps", 16)) == 16
