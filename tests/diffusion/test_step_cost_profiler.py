# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from vllm_omni.diffusion.profiler.step_cost_profiler import DiffusionStepCostProfiler
from vllm_omni.diffusion.worker.utils import DiffusionRequestState


def _make_state(
    req_id: str,
    *,
    step_index: int = 0,
    height: int = 512,
    prompt_count: int = 1,
    outputs: int = 1,
    guidance_scale: float = 0.0,
    true_cfg_scale=None,
    do_classifier_free_guidance: bool = False,
    do_true_cfg: bool = False,
    prompts=None,
) -> DiffusionRequestState:
    sampling = SimpleNamespace(
        height=height,
        width=height,
        num_frames=1,
        num_outputs_per_prompt=outputs,
        do_classifier_free_guidance=do_classifier_free_guidance,
        guidance_scale=guidance_scale,
        true_cfg_scale=true_cfg_scale,
        extra_args={
            "profile_combo_id": f"{height}x{height}_b2",
            "profile_phase": "measure",
            "profile_repeat": 0,
            "profile_batch_size": 2,
            "profile_shape": f"{height}x{height}",
            "profile_mode": "aligned",
            "profile_workload": "current-mix",
        },
    )
    state = DiffusionRequestState(
        req_id=req_id,
        sampling=sampling,
        prompts=prompts if prompts is not None else [f"prompt-{i}" for i in range(prompt_count)],
        latents=torch.zeros(1, 4),
        timesteps=torch.arange(4),
        step_index=step_index,
    )
    state.do_true_cfg = do_true_cfg
    return state


def _make_profiler(tmp_path, monkeypatch):
    output_path = tmp_path / "step_cost_raw.jsonl"
    monkeypatch.setenv("RANK", "0")
    profiler = DiffusionStepCostProfiler.from_od_config(
        SimpleNamespace(
            model="Qwen/Qwen-Image",
            max_num_seqs=4,
            omni_replica_id=0,
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
            additional_config={
                "diffusion_step_profile": {
                    "enabled": True,
                    "output_path": str(output_path),
                    "sync_device": False,
                }
            },
        ),
        device=torch.device("cpu"),
    )
    assert profiler is not None
    return profiler, output_path


def _write_single_record(profiler, state):
    profiler.write_step_record(
        scheduler_output=SimpleNamespace(step_id=1, scheduled_req_ids=[state.req_id]),
        states=[state],
        timings_ms={
            "prepare_batch_ms": 0.0,
            "denoise_ms": 1.0,
            "step_scheduler_ms": 0.0,
            "post_decode_ms": 0.0,
            "total_step_ms": 1.0,
        },
        interrupted=False,
    )


def test_step_cost_profiler_disabled_without_config() -> None:
    profiler = DiffusionStepCostProfiler.from_od_config(
        SimpleNamespace(additional_config={}),
        device=torch.device("cpu"),
    )

    assert profiler is None


def test_step_cost_profiler_records_as_rank_zero_without_rank_env(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "step_cost_raw.jsonl"
    monkeypatch.delenv("RANK", raising=False)

    profiler = DiffusionStepCostProfiler.from_od_config(
        SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
            additional_config={
                "diffusion_step_profile": {
                    "enabled": True,
                    "output_path": str(output_path),
                    "sync_device": False,
                }
            },
        ),
        device=torch.device("cpu"),
    )

    assert profiler is not None
    assert profiler.rank == 0
    assert profiler.enabled is True


def test_step_cost_profiler_writes_jsonl_record(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "step_cost_raw.jsonl"
    monkeypatch.setenv("RANK", "0")
    profiler = DiffusionStepCostProfiler.from_od_config(
        SimpleNamespace(
            model="Qwen/Qwen-Image",
            stage_id=0,
            max_num_seqs=4,
            omni_replica_id=0,
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
            additional_config={
                "diffusion_step_profile": {
                    "enabled": True,
                    "output_path": str(output_path),
                    "sync_device": False,
                }
            },
        ),
        device=torch.device("cpu"),
    )

    assert profiler is not None
    states = [_make_state("a", step_index=0), _make_state("b", step_index=0)]
    scheduler_output = SimpleNamespace(step_id=7, scheduled_req_ids=["a", "b"])
    profiler.write_step_record(
        scheduler_output=scheduler_output,
        states=states,
        timings_ms={
            "prepare_batch_ms": 1.0,
            "denoise_ms": 10.0,
            "step_scheduler_ms": 0.5,
            "post_decode_ms": 0.0,
            "total_step_ms": 12.0,
        },
        interrupted=False,
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["batch_size"] == 2
    assert rows[0]["scheduled_req_ids"] == ["a", "b"]
    assert rows[0]["step_indices"] == [0, 0]
    assert rows[0]["denoise_ms"] == 10.0
    assert rows[0]["post_decode_ms"] == 0.0
    assert rows[0]["shape_key"] == "512x512x1"
    assert rows[0]["replica_id"] == 0
    assert rows[0]["profile_tags"]["profile_combo_id"] == "512x512_b2"
    assert rows[0]["profile_tags"]["profile_workload"] == "current-mix"


def test_step_cost_profiler_counts_multi_prompt_effective_batch(tmp_path, monkeypatch) -> None:
    profiler, output_path = _make_profiler(tmp_path, monkeypatch)

    _write_single_record(
        profiler,
        _make_state("multi", prompt_count=2, outputs=3, guidance_scale=7.5, do_true_cfg=True),
    )

    row = json.loads(output_path.read_text().splitlines()[0])
    assert row["batch_size"] == 1
    assert row["effective_batch_size"] == 12.0


def test_step_cost_profiler_effective_batch_uses_prepared_true_cfg_state(tmp_path, monkeypatch) -> None:
    profiler, output_path = _make_profiler(tmp_path, monkeypatch)

    _write_single_record(
        profiler,
        _make_state(
            "qwen-neg",
            outputs=2,
            true_cfg_scale=None,
            do_true_cfg=True,
            prompts=[{"prompt": "a prompt", "negative_prompt": "bad quality"}],
        ),
    )
    _write_single_record(
        profiler,
        _make_state(
            "qwen-scale-disabled",
            outputs=2,
            true_cfg_scale=1.0,
            do_true_cfg=False,
            prompts=[{"prompt": "a prompt", "negative_prompt": "bad quality"}],
        ),
    )
    _write_single_record(
        profiler,
        _make_state(
            "qwen-no-cfg",
            outputs=2,
            true_cfg_scale=4.0,
            guidance_scale=7.5,
            do_classifier_free_guidance=True,
            do_true_cfg=False,
            prompts=["a prompt"],
        ),
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]

    assert rows[0]["effective_batch_size"] == 4.0
    assert rows[1]["effective_batch_size"] == 2.0
    assert rows[2]["effective_batch_size"] == 2.0


def test_step_cost_profiler_isolates_nonzero_replica_output(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "step_cost_raw.jsonl"
    monkeypatch.setenv("RANK", "0")

    profiler = DiffusionStepCostProfiler.from_od_config(
        SimpleNamespace(
            omni_replica_id=3,
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
            additional_config={
                "diffusion_step_profile": {
                    "enabled": True,
                    "output_path": str(output_path),
                    "sync_device": False,
                }
            },
        ),
        device=torch.device("cpu"),
    )

    assert profiler is not None
    assert profiler.output_path.endswith("step_cost_raw.replica3.jsonl")


def test_step_cost_profiler_default_output_path_is_rank_and_replica_scoped(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "0")
    profiler = DiffusionStepCostProfiler.from_od_config(
        SimpleNamespace(
            omni_replica_id=5,
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
            additional_config={"diffusion_step_profile": {"enabled": True, "sync_device": False}},
        ),
        device=torch.device("cpu"),
    )

    assert profiler is not None
    assert profiler.output_path == "/tmp/vllm_omni_step_cost_raw.replica5.rank0.jsonl"


def test_step_cost_profiler_record_all_adds_rank_suffix_for_explicit_path(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "step_cost_raw.jsonl"
    monkeypatch.setenv("RANK", "1")

    profiler = DiffusionStepCostProfiler.from_od_config(
        SimpleNamespace(
            omni_replica_id=0,
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
            additional_config={
                "diffusion_step_profile": {
                    "enabled": True,
                    "output_path": str(output_path),
                    "record_rank": "all",
                    "sync_device": False,
                }
            },
        ),
        device=torch.device("cpu"),
    )

    assert profiler is not None
    assert profiler.output_path.endswith("step_cost_raw.replica0.rank1.jsonl")


def test_step_cost_profiler_record_rank_mismatch_disables_writer(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "step_cost_raw.jsonl"
    monkeypatch.setenv("RANK", "1")

    profiler = DiffusionStepCostProfiler.from_od_config(
        SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
            additional_config={
                "diffusion_step_profile": {
                    "enabled": True,
                    "output_path": str(output_path),
                    "record_rank": 0,
                    "sync_device": False,
                }
            },
        ),
        device=torch.device("cpu"),
    )

    assert profiler is not None
    assert profiler.enabled is False
    profiler.write_step_record(
        scheduler_output=SimpleNamespace(step_id=1, scheduled_req_ids=["a"]),
        states=[_make_state("a")],
        timings_ms={
            "prepare_batch_ms": 0.0,
            "denoise_ms": 1.0,
            "step_scheduler_ms": 0.0,
            "post_decode_ms": 0.0,
            "total_step_ms": 1.0,
        },
        interrupted=False,
    )
    assert not output_path.exists()
