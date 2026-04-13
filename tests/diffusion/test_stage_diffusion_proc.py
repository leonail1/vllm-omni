# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for StageDiffusionProc request construction and config enrichment."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_process_batch_request_preserves_parent_request_id_and_kv_sender_info():
    """Verify that batched requests keep the parent request metadata intact."""
    async def run_test():
        captured = {}

        def step(request):
            captured["request"] = request
            return [
                SimpleNamespace(
                    images=["img-1"],
                    _multimodal_output={},
                    _custom_output={},
                    metrics={},
                    stage_durations={},
                    peak_memory_mb=0.0,
                    latents=None,
                    trajectory_latents=None,
                    trajectory_timesteps=None,
                    trajectory_log_probs=None,
                    trajectory_decoded=None,
                    final_output_type="image",
                ),
                SimpleNamespace(
                    images=["img-2"],
                    _multimodal_output={},
                    _custom_output={},
                    metrics={},
                    stage_durations={},
                    peak_memory_mb=0.0,
                    latents=None,
                    trajectory_latents=None,
                    trajectory_timesteps=None,
                    trajectory_log_probs=None,
                    trajectory_decoded=None,
                    final_output_type="image",
                ),
            ]

        proc = object.__new__(StageDiffusionProc)
        proc._engine = SimpleNamespace(step=step)
        proc._executor = ThreadPoolExecutor(max_workers=1)

        try:
            result = await proc._process_batch_request(
                request_id="req-parent",
                prompts=["hello", "world"],
                sampling_params_dict=asdict(OmniDiffusionSamplingParams()),
                kv_sender_info={0: {"host": "10.0.0.2", "zmq_port": 50151}},
            )
        finally:
            proc._executor.shutdown(wait=True)

        request = captured["request"]
        assert request.request_id == "req-parent"
        assert request.request_ids == ["req-parent-0", "req-parent-1"]
        assert request.kv_sender_info == {0: {"host": "10.0.0.2", "zmq_port": 50151}}
        assert result.request_id == "req-parent"
        assert result.images == ["img-1", "img-2"]

    asyncio.run(run_test())


def test_enrich_config_routes_s2v_model_type(monkeypatch):
    """Verify that ``model_type=s2v`` resolves to the Wan S2V pipeline."""
    def fake_get_hf_file_to_dict(file_name: str, model: str):
        if file_name == "model_index.json":
            return None
        if file_name == "config.json":
            return {"model_type": "s2v", "_class_name": "WanModel_S2V", "audio_dim": 1024}
        raise AssertionError(f"Unexpected file lookup: {file_name}")

    monkeypatch.setattr(
        "vllm_omni.diffusion.stage_diffusion_proc.get_hf_file_to_dict",
        fake_get_hf_file_to_dict,
    )

    proc = object.__new__(StageDiffusionProc)
    proc._od_config = OmniDiffusionConfig(model="/tmp/fake-wan22-s2v")

    proc._enrich_config()

    assert proc._od_config.model_class_name == "WanS2VPipeline"
    assert proc._od_config.tf_model_config.get("model_type") == "s2v"
