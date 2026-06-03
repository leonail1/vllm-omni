import json
import time
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.distributed.omni_coordinator import ReplicaInfo, ReplicaStatus
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.cpu]


class _FakeDiffusionClient:
    stage_type = "diffusion"
    final_output = True

    def __init__(self, request_address: str, snapshot: dict[str, Any] | None):
        self.request_address = request_address
        self.snapshot = snapshot
        self.rpc_calls = 0

    async def collective_rpc_async(
        self,
        method: str,
        timeout: float | None = None,  # noqa: ARG002
        args: tuple[Any, ...] = (),  # noqa: ARG002
        kwargs: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> Any:
        self.rpc_calls += 1
        assert method == "get_scheduler_load_snapshot"
        return self.snapshot


class _FakeHub:
    def __init__(self, replicas: list[ReplicaInfo]):
        self._replicas = replicas

    def get_replicas_for_stage(self, stage_id: int):
        return type(
            "ReplicaList",
            (),
            {"replicas": [rep for rep in self._replicas if rep.stage_id == stage_id]},
        )()


def _replica(input_addr: str, queue_length: int = 0) -> ReplicaInfo:
    return ReplicaInfo(
        input_addr=input_addr,
        output_addr=input_addr.replace("input", "output"),
        stage_id=0,
        status=ReplicaStatus.UP,
        queue_length=queue_length,
        last_heartbeat=0.0,
        registered_at=0.0,
    )


def _snapshot(
    *,
    safe_admit_capacity: int,
    estimated_step_ms: float,
    min_remaining_steps: int,
    max_remaining_steps: int | None = None,
    num_waiting: int = 0,
    num_running: int = 0,
    key: dict[str, Any] | None = None,
    effective_batch_size: float | None = None,
    min_laxity_ms: float | None = None,
    incremental_step_ms_if_add_one: float | None = None,
) -> dict[str, Any]:
    if effective_batch_size is None:
        effective_batch_size = float(num_running + num_waiting)
    return {
        "policy": "SloStepScheduler",
        "timestamp_s": time.time(),
        "num_waiting": num_waiting,
        "num_running": num_running,
        "max_num_running": 2,
        "safe_admit_capacity": safe_admit_capacity,
        "buckets": [
            {
                "key": key,
                "candidate_batch_size": num_running + num_waiting,
                "effective_batch_size": effective_batch_size,
                "estimated_step_ms": estimated_step_ms,
                "incremental_step_ms_if_add_one": incremental_step_ms_if_add_one,
                "estimated_step_ms_if_add_one": None
                if incremental_step_ms_if_add_one is None
                else estimated_step_ms + incremental_step_ms_if_add_one,
                "min_remaining_steps": min_remaining_steps,
                "max_remaining_steps": min_remaining_steps
                if max_remaining_steps is None
                else max_remaining_steps,
                "min_laxity_ms": min_laxity_ms,
                "shape": {
                    "width": None if key is None else key.get("width"),
                    "height": None if key is None else key.get("height"),
                    "num_frames": 1 if key is None else key.get("num_frames", 1),
                },
            }
        ],
    }


def _task(deadline_offset_s: float = 3.0) -> dict[str, Any]:
    now_s = time.time()
    params = OmniDiffusionSamplingParams(
        height=1024,
        width=1024,
        num_inference_steps=50,
        reference_cost_ms=1000.0,
        slo_ms=3000.0,
        arrival_time_s=now_s,
        deadline_time_s=now_s + deadline_offset_s,
    )
    return {"request_id": "req-slo", "sampling_params": params}


def _cost_model_file(tmp_path) -> str:
    path = tmp_path / "step_cost_model.json"
    path.write_text(
        json.dumps(
            {
                "table_lookup": {
                    "Qwen/Qwen-Image": {
                        "1024x1024x1": {
                            "1": {
                                "1.0": {
                                    "effective_batch_size": 1.0,
                                    "denoise_step_ms": 100.0,
                                    "p90_ms": 100.0,
                                },
                                "2.0": {
                                    "effective_batch_size": 2.0,
                                    "denoise_step_ms": 300.0,
                                    "p90_ms": 300.0,
                                }
                            },
                            "2": {
                                "2.0": {
                                    "effective_batch_size": 2.0,
                                    "denoise_step_ms": 200.0,
                                    "p90_ms": 200.0,
                                }
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _stage_config(cost_model_path: str) -> Any:
    return SimpleNamespace(
        model="Qwen/Qwen-Image",
        additional_config={
            "diffusion_slo_scheduler": {
                "enable_stagepool_slo": True,
                "step_cost_model_path": cost_model_path,
                "step_cost_metric": "p90_ms",
            }
        },
    )


def _stage_config_with_slo_options(options: dict[str, Any]) -> Any:
    slo_options = {"enable_stagepool_slo": True, **options}
    return SimpleNamespace(
        model="Qwen/Qwen-Image",
        additional_config={"diffusion_slo_scheduler": slo_options},
    )


def _qwen_dynamic_stage_config(
    options: dict[str, Any] | None = None,
    *,
    policy: str = "slo_no_preemption_token_guarded",
) -> Any:
    slo_options = {"enable_stagepool_slo": True, **(options or {})}
    return SimpleNamespace(
        model="Qwen/Qwen-Image",
        model_class_name="QwenImagePipeline",
        step_execution=True,
        enforce_eager=True,
        cache_backend="none",
        diffusion_kv_cache_dtype=None,
        parallel_config=SimpleNamespace(
            sequence_parallel_size=1,
            ring_degree=1,
            cfg_parallel_size=1,
            use_hsdp=False,
        ),
        additional_config={
            "diffusion_dynamic_step_batching_enabled": True,
            "diffusion_scheduler_policy": policy,
            "diffusion_slo_scheduler": slo_options,
        },
    )


class _RecordingCostModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def latent_tokens(self, width: float | int | None, height: float | int | None, num_frames: float | int | None = 1) -> int | None:
        if width is None or height is None:
            return None
        return max(int(round((float(width) / 16.0) * (float(height) / 16.0) * max(float(num_frames or 1), 1.0))), 1)

    def estimate(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            step_ms=max(float(kwargs.get("effective_batch_size", 1.0)) * 100.0, 0.001),
            source="recording",
            latent_tokens=self.latent_tokens(
                kwargs.get("width"), kwargs.get("height"), kwargs.get("num_frames", 1)
            ),
        )


def _stage_config_with_missing_model(cost_model_path: str) -> Any:
    return SimpleNamespace(
        model="Missing/Model",
        additional_config={
            "diffusion_slo_scheduler": {
                "enable_stagepool_slo": True,
                "step_cost_model_path": cost_model_path,
                "step_cost_metric": "p90_ms",
            }
        },
    )


@pytest.mark.asyncio
async def test_stage_pool_local_slo_routing_prefers_replica_with_lower_admit_delay() -> None:
    busy_addr = "tcp://host-a:1000/input"
    idle_addr = "tcp://host-b:1000/input"
    pool = StagePool(
        0,
        [
            _FakeDiffusionClient(
                busy_addr,
                _snapshot(
                    safe_admit_capacity=0,
                    estimated_step_ms=120.0,
                    min_remaining_steps=20,
                    num_waiting=4,
                    num_running=2,
                ),
            ),
            _FakeDiffusionClient(
                idle_addr,
                _snapshot(
                    safe_admit_capacity=1,
                    estimated_step_ms=80.0,
                    min_remaining_steps=1,
                    num_waiting=0,
                    num_running=1,
                ),
            ),
        ],
        stage_vllm_config=_stage_config_with_slo_options({}),
    )

    assert await pool._select_slo_aware_local_replica_id(_task()) == 1


@pytest.mark.asyncio
async def test_stage_pool_local_slo_routing_round_robins_tied_replicas() -> None:
    snapshot = _snapshot(safe_admit_capacity=1, estimated_step_ms=80.0, min_remaining_steps=1)
    pool = StagePool(
        0,
        [
            _FakeDiffusionClient("tcp://host-a:1000/input", snapshot),
            _FakeDiffusionClient("tcp://host-b:1000/input", snapshot),
        ],
        stage_vllm_config=_stage_config_with_slo_options({}),
    )

    assert await pool._select_slo_aware_local_replica_id(_task()) == 0
    assert await pool._select_slo_aware_local_replica_id(_task()) == 1
    assert await pool._select_slo_aware_local_replica_id(_task()) == 0


@pytest.mark.asyncio
async def test_stage_pool_distributed_slo_routing_returns_candidate_index() -> None:
    addr0 = "tcp://host-a:1000/input"
    addr1 = "tcp://host-b:1000/input"
    pool = StagePool(
        0,
        [
            _FakeDiffusionClient(addr0, _snapshot(safe_admit_capacity=0, estimated_step_ms=100.0, min_remaining_steps=10)),
            _FakeDiffusionClient(addr1, _snapshot(safe_admit_capacity=1, estimated_step_ms=100.0, min_remaining_steps=1)),
        ],
        stage_vllm_config=_stage_config_with_slo_options({}),
    )
    candidates = [(_replica(addr0, queue_length=10), 0), (_replica(addr1, queue_length=1), 1)]

    assert await pool._select_slo_aware_candidate_index(candidates, _task()) == 1


@pytest.mark.asyncio
async def test_stage_pool_distributed_slo_routing_round_robins_tied_candidates() -> None:
    addr0 = "tcp://host-a:1000/input"
    addr1 = "tcp://host-b:1000/input"
    snapshot = _snapshot(safe_admit_capacity=1, estimated_step_ms=100.0, min_remaining_steps=1)
    pool = StagePool(
        0,
        [
            _FakeDiffusionClient(addr0, snapshot),
            _FakeDiffusionClient(addr1, snapshot),
        ],
        stage_vllm_config=_stage_config_with_slo_options({}),
    )
    candidates = [(_replica(addr0, queue_length=0), 0), (_replica(addr1, queue_length=0), 1)]

    assert await pool._select_slo_aware_candidate_index(candidates, _task()) == 0
    assert await pool._select_slo_aware_candidate_index(candidates, _task()) == 1
    assert await pool._select_slo_aware_candidate_index(candidates, _task()) == 0


@pytest.mark.asyncio
async def test_stage_pool_slo_routing_falls_back_when_snapshots_are_unsupported() -> None:
    pool = StagePool(
        0,
        [
            _FakeDiffusionClient("tcp://host-a:1000/input", {"supported": False}),
            _FakeDiffusionClient("tcp://host-b:1000/input", {"malformed": True}),
        ],
    )

    assert await pool._select_slo_aware_local_replica_id(_task()) is None


@pytest.mark.asyncio
async def test_stage_pool_slo_routing_ignores_non_slo_scheduler_snapshots() -> None:
    pool = StagePool(
        0,
        [
            _FakeDiffusionClient(
                "tcp://host-a:1000/input",
                {
                    "policy": "StepScheduler",
                    "timestamp_s": time.time(),
                    "num_waiting": 0,
                    "num_running": 0,
                    "safe_admit_capacity": 1,
                    "buckets": [],
                },
            ),
            _FakeDiffusionClient("tcp://host-b:1000/input", {"supported": False}),
        ],
    )

    assert await pool._select_slo_aware_local_replica_id(_task()) is None


@pytest.mark.asyncio
async def test_stage_pool_scheduler_snapshot_cache_reuses_recent_rpc_result() -> None:
    client = _FakeDiffusionClient(
        "tcp://host-a:1000/input",
        _snapshot(safe_admit_capacity=1, estimated_step_ms=50.0, min_remaining_steps=1),
    )
    pool = StagePool(0, [client])

    first = await pool._get_scheduler_snapshot(0)
    second = await pool._get_scheduler_snapshot(0)

    assert first is second
    assert client.rpc_calls == 1


@pytest.mark.asyncio
async def test_stage_pool_scheduler_snapshot_accepts_cross_machine_timestamp_skew() -> None:
    snapshot = _snapshot(safe_admit_capacity=1, estimated_step_ms=50.0, min_remaining_steps=1)
    snapshot["timestamp_s"] = 0.0
    pool = StagePool(0, [_FakeDiffusionClient("tcp://host-a:1000/input", snapshot)])

    assert await pool._get_scheduler_snapshot(0) == snapshot


@pytest.mark.asyncio
async def test_stage_pool_laxity_window_packs_same_key_when_safe() -> None:
    task = _task(deadline_offset_s=3.0)
    key = StagePool._sampling_key_dict(task["sampling_params"])
    pool = StagePool(
        0,
        [
            _FakeDiffusionClient(
                "tcp://host-a:1000/input",
                _snapshot(
                    safe_admit_capacity=1,
                    estimated_step_ms=100.0,
                    min_remaining_steps=10,
                    num_running=1,
                    key=key,
                    min_laxity_ms=5000.0,
                ),
            ),
            _FakeDiffusionClient(
                "tcp://host-b:1000/input",
                {
                    "policy": "SloStepScheduler",
                    "timestamp_s": time.time(),
                    "num_waiting": 0,
                    "num_running": 0,
                    "max_num_running": 2,
                    "safe_admit_capacity": 1,
                    "buckets": [],
                },
            ),
        ],
        stage_vllm_config=_stage_config_with_slo_options({"stagepool_laxity_window_ms": 500.0}),
    )

    assert await pool._select_slo_aware_local_replica_id(task) == 0


@pytest.mark.asyncio
async def test_stage_pool_laxity_window_keeps_much_safer_replica() -> None:
    task = _task(deadline_offset_s=3.0)
    key = StagePool._sampling_key_dict(task["sampling_params"])
    pool = StagePool(
        0,
        [
            _FakeDiffusionClient(
                "tcp://host-a:1000/input",
                _snapshot(
                    safe_admit_capacity=1,
                    estimated_step_ms=1200.0,
                    min_remaining_steps=10,
                    num_running=1,
                    key=key,
                    min_laxity_ms=10000.0,
                ),
            ),
            _FakeDiffusionClient(
                "tcp://host-b:1000/input",
                {
                    "policy": "SloStepScheduler",
                    "timestamp_s": time.time(),
                    "num_waiting": 0,
                    "num_running": 0,
                    "max_num_running": 2,
                    "safe_admit_capacity": 1,
                    "buckets": [],
                },
            ),
        ],
        stage_vllm_config=_stage_config_with_slo_options({"stagepool_laxity_window_ms": 500.0}),
    )

    assert await pool._select_slo_aware_local_replica_id(task) == 1


@pytest.mark.asyncio
async def test_stage_pool_pack_min_laxity_blocks_low_slack_packing() -> None:
    task = _task(deadline_offset_s=3.0)
    key = StagePool._sampling_key_dict(task["sampling_params"])
    pool = StagePool(
        0,
        [
            _FakeDiffusionClient(
                "tcp://host-a:1000/input",
                _snapshot(
                    safe_admit_capacity=1,
                    estimated_step_ms=1100.0,
                    min_remaining_steps=10,
                    num_running=1,
                    key=key,
                    min_laxity_ms=10000.0,
                ),
            ),
            _FakeDiffusionClient(
                "tcp://host-b:1000/input",
                {
                    "policy": "SloStepScheduler",
                    "timestamp_s": time.time(),
                    "num_waiting": 0,
                    "num_running": 0,
                    "max_num_running": 2,
                    "safe_admit_capacity": 1,
                    "buckets": [],
                },
            ),
        ],
        stage_vllm_config=_stage_config_with_slo_options(
            {
                "stagepool_laxity_window_ms": 1500.0,
                "stagepool_pack_min_laxity_ms": 1000.0,
            }
        ),
    )

    assert await pool._select_slo_aware_local_replica_id(task) == 1


@pytest.mark.asyncio
async def test_stage_pool_pack_queue_guard_blocks_busy_packing_replica() -> None:
    task = _task(deadline_offset_s=3.0)
    key = StagePool._sampling_key_dict(task["sampling_params"])
    pool = StagePool(
        0,
        [
            _FakeDiffusionClient(
                "tcp://host-a:1000/input",
                _snapshot(
                    safe_admit_capacity=1,
                    estimated_step_ms=100.0,
                    min_remaining_steps=10,
                    num_waiting=1,
                    num_running=1,
                    key=key,
                    min_laxity_ms=5000.0,
                ),
            ),
            _FakeDiffusionClient(
                "tcp://host-b:1000/input",
                {
                    "policy": "SloStepScheduler",
                    "timestamp_s": time.time(),
                    "num_waiting": 0,
                    "num_running": 0,
                    "max_num_running": 2,
                    "safe_admit_capacity": 1,
                    "buckets": [],
                },
            ),
        ],
        stage_vllm_config=_stage_config_with_slo_options(
            {
                "stagepool_laxity_window_ms": 500.0,
                "stagepool_pack_max_queue_length": 2,
            }
        ),
    )

    assert await pool._select_slo_aware_local_replica_id(task) == 1


@pytest.mark.asyncio
async def test_stage_pool_pack_bucket_size_guard_blocks_large_bucket() -> None:
    task = _task(deadline_offset_s=3.0)
    key = StagePool._sampling_key_dict(task["sampling_params"])
    pool = StagePool(
        0,
        [
            _FakeDiffusionClient(
                "tcp://host-a:1000/input",
                _snapshot(
                    safe_admit_capacity=1,
                    estimated_step_ms=100.0,
                    min_remaining_steps=10,
                    num_running=2,
                    key=key,
                    min_laxity_ms=5000.0,
                ),
            ),
            _FakeDiffusionClient(
                "tcp://host-b:1000/input",
                {
                    "policy": "SloStepScheduler",
                    "timestamp_s": time.time(),
                    "num_waiting": 0,
                    "num_running": 0,
                    "max_num_running": 2,
                    "safe_admit_capacity": 1,
                    "buckets": [],
                },
            ),
        ],
        stage_vllm_config=_stage_config_with_slo_options(
            {
                "stagepool_laxity_window_ms": 500.0,
                "stagepool_pack_max_matching_bucket_size": 2,
            }
        ),
    )

    assert await pool._select_slo_aware_local_replica_id(task) == 1


@pytest.mark.asyncio
async def test_stage_pool_cost_model_avoids_replica_that_would_hurt_existing_bucket(tmp_path) -> None:
    task = _task(deadline_offset_s=10.0)
    key = StagePool._sampling_key_dict(task["sampling_params"])
    cost_model_path = _cost_model_file(tmp_path)
    pool = StagePool(
        0,
        [
            _FakeDiffusionClient(
                "tcp://host-a:1000/input",
                _snapshot(
                    safe_admit_capacity=1,
                    estimated_step_ms=100.0,
                    min_remaining_steps=50,
                    num_running=1,
                    key=key,
                    effective_batch_size=1.0,
                    min_laxity_ms=20.0,
                ),
            ),
            _FakeDiffusionClient(
                "tcp://host-b:1000/input",
                _snapshot(
                    safe_admit_capacity=1,
                    estimated_step_ms=100.0,
                    min_remaining_steps=1,
                    num_running=0,
                    key=None,
                    effective_batch_size=0.0,
                ),
            ),
        ],
        stage_vllm_config=_stage_config(cost_model_path),
    )

    assert await pool._select_slo_aware_local_replica_id(task) == 1


@pytest.mark.asyncio
async def test_stage_pool_qwen_dynamic_key_ignores_shape_for_token_policy() -> None:
    now_s = time.time()
    incoming_params = OmniDiffusionSamplingParams(
        height=1024,
        width=1024,
        num_inference_steps=50,
        reference_cost_ms=1000.0,
        slo_ms=3000.0,
        arrival_time_s=now_s,
        deadline_time_s=now_s + 3.0,
        true_cfg_scale=1.0,
    )
    bucket_params = OmniDiffusionSamplingParams(height=512, width=512, true_cfg_scale=1.0)
    dynamic_bucket_key = StagePool._sampling_key_dict(bucket_params, dynamic_qwen=True)
    task = {"request_id": "req-slo", "sampling_params": incoming_params}
    pool = StagePool(
        0,
        [
            _FakeDiffusionClient(
                "tcp://host-a:1000/input",
                _snapshot(
                    safe_admit_capacity=1,
                    estimated_step_ms=100.0,
                    min_remaining_steps=10,
                    num_running=1,
                    key=dynamic_bucket_key,
                    min_laxity_ms=5000.0,
                ),
            ),
            _FakeDiffusionClient("tcp://host-b:1000/input", {"policy": "TokenSloStepScheduler", "timestamp_s": time.time(), "num_waiting": 0, "num_running": 0, "max_num_running": 2, "safe_admit_capacity": 1, "buckets": []}),
        ],
        stage_vllm_config=_qwen_dynamic_stage_config({"stagepool_laxity_window_ms": 500.0}),
    )

    assert pool._task_sampling_key_dict(task) == dynamic_bucket_key
    assert await pool._select_slo_aware_local_replica_id(task) == 0


def test_stage_pool_token_incoming_step_uses_max_token_bucket_shape() -> None:
    incoming_params = OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=50,
        true_cfg_scale=1.0,
    )
    bucket_key = StagePool._sampling_key_dict(
        OmniDiffusionSamplingParams(height=1024, width=1024, true_cfg_scale=1.0),
        dynamic_qwen=True,
    )
    pool = StagePool(
        0,
        [_FakeDiffusionClient("tcp://host-a:1000/input", None)],
        stage_vllm_config=_qwen_dynamic_stage_config(),
    )
    recorder = _RecordingCostModel()
    pool._slo_cost_model = recorder
    snapshot = {
        "buckets": [
            {
                "key": bucket_key,
                "candidate_batch_size": 1,
                "effective_batch_size": 1.0,
                "max_latent_tokens": 4096,
                "total_token_work": 4096.0,
                "shape": {"width": 1024, "height": 1024, "num_frames": 1},
            }
        ]
    }

    pool._estimate_incoming_step(incoming_params, snapshot, has_negative_prompt=False)

    assert recorder.calls[-1]["width"] == 1024
    assert recorder.calls[-1]["height"] == 1024
    assert recorder.calls[-1]["batch_size"] == 2
    assert recorder.calls[-1]["effective_batch_size"] == pytest.approx(1.25)


def test_stage_pool_existing_laxity_uses_token_weighted_increment() -> None:
    now_s = time.time()
    incoming_params = OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=50,
        arrival_time_s=now_s,
        deadline_time_s=now_s + 10.0,
        true_cfg_scale=1.0,
    )
    bucket_key = StagePool._sampling_key_dict(incoming_params, dynamic_qwen=True)
    pool = StagePool(0, [_FakeDiffusionClient("tcp://host-a:1000/input", None)], stage_vllm_config=_qwen_dynamic_stage_config())
    recorder = _RecordingCostModel()
    pool._slo_cost_model = recorder
    snapshot = _snapshot(safe_admit_capacity=1, estimated_step_ms=100.0, min_remaining_steps=5, num_running=1, key=bucket_key, min_laxity_ms=1000.0)
    snapshot["buckets"][0].update({"max_latent_tokens": 4096, "total_token_work": 4096.0, "shape": {"width": 1024, "height": 1024, "num_frames": 1}})

    pool._estimate_existing_laxity_after_ms({"sampling_params": incoming_params}, snapshot)

    assert recorder.calls[-1]["width"] == 1024
    assert recorder.calls[-1]["height"] == 1024
    assert recorder.calls[-1]["effective_batch_size"] == pytest.approx(1.25)


def test_stage_pool_existing_laxity_token_fallback_uses_incoming_work() -> None:
    now_s = time.time()
    incoming_params = OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=50,
        arrival_time_s=now_s,
        deadline_time_s=now_s + 10.0,
        true_cfg_scale=1.0,
    )
    bucket_key = StagePool._sampling_key_dict(incoming_params, dynamic_qwen=True)
    pool = StagePool(
        0,
        [_FakeDiffusionClient("tcp://host-a:1000/input", None)],
        stage_vllm_config=_qwen_dynamic_stage_config({"batch_growth_alpha": 0.6}),
    )
    pool._slo_cost_model = None
    snapshot = _snapshot(
        safe_admit_capacity=1,
        estimated_step_ms=100.0,
        incremental_step_ms_if_add_one=10.0,
        min_remaining_steps=5,
        num_running=1,
        key=bucket_key,
        min_laxity_ms=1000.0,
    )
    snapshot["buckets"][0].update(
        {
            "max_latent_tokens": 4096,
            "total_token_work": 4096.0,
            "shape": {"width": 1024, "height": 1024, "num_frames": 1},
        }
    )

    assert pool._estimate_existing_laxity_after_ms({"sampling_params": incoming_params}, snapshot) == pytest.approx(
        925.0
    )


def test_stage_pool_token_fallback_preserves_zero_batch_growth_alpha() -> None:
    now_s = time.time()
    incoming_params = OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=50,
        arrival_time_s=now_s,
        deadline_time_s=now_s + 10.0,
        true_cfg_scale=1.0,
    )
    bucket_key = StagePool._sampling_key_dict(incoming_params, dynamic_qwen=True)
    pool = StagePool(
        0,
        [_FakeDiffusionClient("tcp://host-a:1000/input", None)],
        stage_vllm_config=_qwen_dynamic_stage_config({"batch_growth_alpha": 0.0}),
    )
    pool._slo_cost_model = None
    snapshot = _snapshot(
        safe_admit_capacity=1,
        estimated_step_ms=100.0,
        incremental_step_ms_if_add_one=10.0,
        min_remaining_steps=5,
        num_running=1,
        key=bucket_key,
        min_laxity_ms=1000.0,
    )
    snapshot["buckets"][0].update(
        {
            "max_latent_tokens": 4096,
            "total_token_work": 4096.0,
            "shape": {"width": 1024, "height": 1024, "num_frames": 1},
        }
    )

    assert pool._slo_batch_growth_alpha == pytest.approx(0.0)
    assert pool._estimate_existing_laxity_after_ms({"sampling_params": incoming_params}, snapshot) == pytest.approx(
        1000.0
    )


@pytest.mark.asyncio
async def test_stage_pool_accepts_adaptive_token_snapshot_and_default_enable() -> None:
    now_s = time.time()
    params = OmniDiffusionSamplingParams(
        height=1024,
        width=1024,
        num_inference_steps=50,
        reference_cost_ms=1000.0,
        slo_ms=3000.0,
        arrival_time_s=now_s,
        deadline_time_s=now_s + 3.0,
        true_cfg_scale=1.0,
    )
    key = StagePool._sampling_key_dict(params, dynamic_qwen=True)
    snapshot = _snapshot(
        safe_admit_capacity=1,
        estimated_step_ms=100.0,
        min_remaining_steps=5,
        num_running=1,
        key=key,
        min_laxity_ms=1000.0,
    )
    snapshot.update({"policy": "AdaptiveTokenSloStepScheduler", "token_pressure": 4096.0})
    snapshot["buckets"][0].update(
        {
            "max_latent_tokens": 4096,
            "total_token_work": 4096.0,
            "token_batch_utilization": 1.0,
            "min_laxity_ratio": 0.5,
        }
    )
    pool = StagePool(
        0,
        [
            _FakeDiffusionClient("tcp://host-a:1000/input", snapshot),
            _FakeDiffusionClient(
                "tcp://host-b:1000/input",
                {
                    "policy": "AdaptiveTokenSloStepScheduler",
                    "timestamp_s": time.time(),
                    "num_waiting": 0,
                    "num_running": 0,
                    "max_num_running": 2,
                    "safe_admit_capacity": 1,
                    "buckets": [],
                },
            ),
        ],
        stage_vllm_config=_qwen_dynamic_stage_config(
            {"stagepool_laxity_window_ms": 500.0},
            policy="slo_no_preemption_token_adaptive",
        ),
    )

    selected = await pool._select_slo_aware_local_replica_id({"request_id": "req", "sampling_params": params})

    assert pool._enable_stagepool_slo is True
    assert selected == 1


def test_stage_pool_adaptive_candidate_profile_records_token_fields() -> None:
    now_s = time.time()
    params = OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=50,
        reference_cost_ms=1000.0,
        slo_ms=3000.0,
        arrival_time_s=now_s,
        deadline_time_s=now_s + 3.0,
        true_cfg_scale=1.0,
    )
    key = StagePool._sampling_key_dict(params, dynamic_qwen=True)
    snapshot = _snapshot(
        safe_admit_capacity=1,
        estimated_step_ms=100.0,
        min_remaining_steps=5,
        num_running=1,
        key=key,
        min_laxity_ms=1000.0,
    )
    snapshot.update({"policy": "AdaptiveTokenSloStepScheduler", "token_pressure": 4096.0})
    snapshot["buckets"][0].update(
        {
            "max_latent_tokens": 4096,
            "total_token_work": 4096.0,
            "token_batch_utilization": 1.0,
            "min_laxity_ratio": 0.5,
        }
    )
    pool = StagePool(
        0,
        [_FakeDiffusionClient("tcp://host-a:1000/input", snapshot)],
        stage_vllm_config=_qwen_dynamic_stage_config(policy="slo_no_preemption_token_adaptive"),
    )

    candidate = pool._stagepool_slo_candidate_profile(
        {"request_id": "req", "sampling_params": params},
        snapshot,
        incoming_key=key,
        now_s=now_s,
        candidate_index=0,
        replica_id=0,
        tie_rank=0,
        fallback_queue_length=0,
    )

    assert candidate["incoming_latent_tokens"] == pytest.approx(1024.0)
    assert candidate["incoming_remaining_steps"] == 50
    assert candidate["matching_bucket_token_work"] == pytest.approx(4096.0)
    assert candidate["token_pressure_after"] > candidate["token_pressure_before"]
    assert candidate["candidate_min_laxity_ratio"] == pytest.approx(0.5)


def test_stage_pool_token_objective_prefers_lower_slo_risk_over_laxity_headroom() -> None:
    pool = StagePool(
        0,
        [_FakeDiffusionClient("tcp://host-a:1000/input", None)],
        stage_vllm_config=_qwen_dynamic_stage_config(
            {
                "stagepool_selection_objective": "token_slo_objective",
                "stagepool_objective_miss_risk_weight": 10.0,
                "stagepool_objective_token_pressure_weight": 0.01,
                "stagepool_objective_queue_weight": 1.0,
                "stagepool_objective_pack_bonus": 1.0,
                "stagepool_objective_safe_capacity_bonus": 1.0,
            },
            policy="slo_no_preemption_token_objective",
        ),
    )
    high_laxity_but_risky = {
        "candidate_index": 0,
        "replica_id": 0,
        "predicted_laxity_ms": 1000.0,
        "incoming_laxity_ms": -50.0,
        "existing_min_laxity_after_ms": 100.0,
        "safe_capacity": 1,
        "queue_length": 0,
        "can_pack_same_key": True,
        "matching_bucket_size": 3,
        "token_pressure_after": 1024.0,
        "tie_rank": 0,
    }
    lower_laxity_but_safe = {
        "candidate_index": 1,
        "replica_id": 1,
        "predicted_laxity_ms": 500.0,
        "incoming_laxity_ms": 100.0,
        "existing_min_laxity_after_ms": 100.0,
        "safe_capacity": 1,
        "queue_length": 0,
        "can_pack_same_key": False,
        "matching_bucket_size": 0,
        "token_pressure_after": 1024.0,
        "tie_rank": 1,
    }

    selected = pool._choose_stagepool_slo_candidate([high_laxity_but_risky, lower_laxity_but_safe])

    assert selected["replica_id"] == 1
    assert high_laxity_but_risky["stagepool_objective_miss_risk_ms"] == pytest.approx(50.0)
    assert lower_laxity_but_safe["stagepool_objective_miss_risk_ms"] == pytest.approx(0.0)
    assert selected["stagepool_selection_objective"] == "token_slo_objective"


def test_stage_pool_legacy_selection_ignores_token_pressure_tie_break() -> None:
    pool = StagePool(
        0,
        [_FakeDiffusionClient("tcp://host-a:1000/input", None)],
        stage_vllm_config=_qwen_dynamic_stage_config(policy="slo_no_preemption_token_guarded"),
    )
    lower_tie_rank_high_pressure = {
        "candidate_index": 0,
        "replica_id": 0,
        "predicted_laxity_ms": 1000.0,
        "safe_capacity": 1,
        "queue_length": 0,
        "can_pack_same_key": False,
        "matching_bucket_size": 0,
        "token_pressure_after": 8192.0,
        "tie_rank": 0,
    }
    higher_tie_rank_low_pressure = {
        "candidate_index": 1,
        "replica_id": 1,
        "predicted_laxity_ms": 1000.0,
        "safe_capacity": 1,
        "queue_length": 0,
        "can_pack_same_key": False,
        "matching_bucket_size": 0,
        "token_pressure_after": 0.0,
        "tie_rank": 1,
    }

    selected = pool._choose_stagepool_slo_candidate(
        [lower_tie_rank_high_pressure, higher_tie_rank_low_pressure]
    )

    assert selected["replica_id"] == 0
    assert "stagepool_objective_score" not in lower_tie_rank_high_pressure
    assert "stagepool_objective_score" not in higher_tie_rank_low_pressure


def test_stage_pool_token_objective_profile_records_score() -> None:
    now_s = time.time()
    params = OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=50,
        reference_cost_ms=1000.0,
        slo_ms=3000.0,
        arrival_time_s=now_s,
        deadline_time_s=now_s + 3.0,
        true_cfg_scale=1.0,
    )
    key = StagePool._sampling_key_dict(params, dynamic_qwen=True)
    snapshot = _snapshot(
        safe_admit_capacity=1,
        estimated_step_ms=100.0,
        min_remaining_steps=5,
        num_running=1,
        key=key,
        min_laxity_ms=1000.0,
    )
    snapshot.update({"policy": "TokenSloStepScheduler", "token_pressure": 4096.0})
    snapshot["buckets"][0].update(
        {
            "max_latent_tokens": 4096,
            "total_token_work": 4096.0,
            "token_batch_utilization": 1.0,
        }
    )
    pool = StagePool(
        0,
        [_FakeDiffusionClient("tcp://host-a:1000/input", snapshot)],
        stage_vllm_config=_qwen_dynamic_stage_config(
            {"stagepool_selection_objective": "token_slo_objective"},
            policy="slo_no_preemption_token_objective",
        ),
    )

    candidate = pool._stagepool_slo_candidate_profile(
        {"request_id": "req", "sampling_params": params},
        snapshot,
        incoming_key=key,
        now_s=now_s,
        candidate_index=0,
        replica_id=0,
        tie_rank=0,
        fallback_queue_length=0,
    )

    assert candidate["stagepool_selection_objective"] == "token_slo_objective"
    assert candidate["stagepool_objective_score"] == pytest.approx(
        candidate["stagepool_objective_miss_risk_ms"]
        + candidate["stagepool_objective_token_pressure_units"] * 0.01
        + candidate["stagepool_objective_queue_cost"]
        - candidate["stagepool_objective_pack_bonus"]
        - candidate["stagepool_objective_safe_capacity_bonus"]
    )


@pytest.mark.parametrize(
    "policy",
    [
        "slo_no_preemption_token_objective",
        "slo_token_stagepool_objective",
    ],
)
def test_stage_pool_objective_policy_defaults_to_objective_mode(policy: str) -> None:
    pool = StagePool(
        0,
        [_FakeDiffusionClient("tcp://host-a:1000/input", None)],
        stage_vllm_config=_qwen_dynamic_stage_config(policy=policy),
    )

    assert pool._slo_stagepool_selection_objective == "token_slo_objective"
    candidate = {
        "candidate_index": 0,
        "replica_id": 0,
        "predicted_laxity_ms": 1000.0,
        "incoming_laxity_ms": 1000.0,
        "existing_min_laxity_after_ms": 1000.0,
        "safe_capacity": 1,
        "queue_length": 0,
        "can_pack_same_key": True,
        "matching_bucket_size": 1,
        "token_pressure_after": 1024.0,
        "tie_rank": 0,
    }

    pool._choose_stagepool_slo_candidate([candidate])

    assert candidate["stagepool_selection_objective"] == "token_slo_objective"
    assert "stagepool_objective_score" in candidate


def test_stage_pool_explicit_legacy_overrides_objective_policy_default() -> None:
    pool = StagePool(
        0,
        [_FakeDiffusionClient("tcp://host-a:1000/input", None)],
        stage_vllm_config=_qwen_dynamic_stage_config(
            {"stagepool_selection_objective": "legacy"},
            policy="slo_token_stagepool_objective",
        ),
    )

    assert pool._slo_stagepool_selection_objective == "legacy"


def test_stage_pool_rejects_unknown_selection_objective() -> None:
    with pytest.raises(ValueError, match="stagepool_selection_objective"):
        StagePool(
            0,
            [_FakeDiffusionClient("tcp://host-a:1000/input", None)],
            stage_vllm_config=_qwen_dynamic_stage_config(
                {"stagepool_selection_objective": "mystery"},
                policy="slo_no_preemption_token_objective",
            ),
        )


def test_stage_pool_accepts_token_step_preemptive_snapshot() -> None:
    snapshot = _snapshot(
        safe_admit_capacity=1,
        estimated_step_ms=100.0,
        min_remaining_steps=5,
        num_running=1,
    )
    snapshot["policy"] = "TokenStepPreemptiveSloStepScheduler"
    snapshot["enable_step_preemption"] = True

    assert StagePool._is_valid_scheduler_snapshot(snapshot) is True


def test_stage_pool_profile_record_includes_selection_objective(tmp_path) -> None:
    profile_path = tmp_path / "stagepool.jsonl"
    pool = StagePool(
        0,
        [_FakeDiffusionClient("tcp://host-a:1000/input", None)],
        stage_vllm_config=_qwen_dynamic_stage_config(
            {"stagepool_selection_objective": "token_slo_objective"},
            policy="slo_no_preemption_token_objective",
        ),
    )
    pool._stagepool_profile_enabled = True
    pool._stagepool_profile_path = str(profile_path)

    pool._write_stagepool_profile("local_slo_select", None, 0, [], time.time())

    record = json.loads(profile_path.read_text(encoding="utf-8").strip())
    assert record["stagepool_selection_objective"] == "token_slo_objective"


def test_stage_pool_profile_only_reject_records_reason_without_rejecting(tmp_path) -> None:
    profile_path = tmp_path / "stagepool.jsonl"
    pool = StagePool(
        0,
        [_FakeDiffusionClient("tcp://host-a:1000/input", None)],
        stage_vllm_config=_qwen_dynamic_stage_config(
            {
                "stagepool_profile_reject_laxity_ms": 0.0,
            },
            policy="slo_no_preemption_token_adaptive",
        ),
    )
    pool._stagepool_profile_enabled = True
    pool._stagepool_profile_path = str(profile_path)
    candidates = [
        {"replica_id": 0, "predicted_laxity_ms": -10.0},
        {"replica_id": 1, "predicted_laxity_ms": -1.0},
    ]

    pool._write_stagepool_profile("local_slo_select", None, 0, candidates, time.time())

    record = json.loads(profile_path.read_text(encoding="utf-8").strip())
    assert record["selected_replica_id"] == 0
    assert record["profile_only_reject_reason"] == "all_replicas_predicted_below_profile_threshold"
    assert all(
        candidate["profile_only_reject_reason"] == "all_replicas_predicted_below_profile_threshold"
        for candidate in record["candidates"]
    )


def test_stage_pool_cost_model_default_source_falls_back_to_reference_cost(tmp_path) -> None:
    task = _task(deadline_offset_s=10.0)
    pool = StagePool(
        0,
        [_FakeDiffusionClient("tcp://host-a:1000/input", _snapshot(safe_admit_capacity=1, estimated_step_ms=100.0, min_remaining_steps=1))],
        stage_vllm_config=_stage_config_with_missing_model(_cost_model_file(tmp_path)),
    )

    assert pool._estimate_task_remaining_cost_ms(task, {"buckets": []}) == pytest.approx(1000.0)


def test_stage_pool_cost_model_effective_batch_counts_prompts(tmp_path) -> None:
    task = _task(deadline_offset_s=10.0)
    task["prompt_count"] = 2
    pool = StagePool(
        0,
        [
            _FakeDiffusionClient(
                "tcp://host-a:1000/input",
                _snapshot(safe_admit_capacity=1, estimated_step_ms=100.0, min_remaining_steps=1),
            )
        ],
        stage_vllm_config=_stage_config(_cost_model_file(tmp_path)),
    )

    assert pool._estimate_task_remaining_cost_ms(task, {"buckets": []}) == pytest.approx(50 * 300.0)


def test_stage_pool_existing_laxity_uses_snapshot_increment_when_cost_model_misses() -> None:
    task = _task(deadline_offset_s=10.0)
    key = StagePool._sampling_key_dict(task["sampling_params"])
    pool = StagePool(0, [_FakeDiffusionClient("tcp://host-a:1000/input", None)])
    snapshot = _snapshot(
        safe_admit_capacity=1,
        estimated_step_ms=100.0,
        incremental_step_ms_if_add_one=10.0,
        min_remaining_steps=5,
        num_running=1,
        key=key,
        effective_batch_size=1.0,
        min_laxity_ms=80.0,
    )

    assert pool._estimate_existing_laxity_after_ms(task, snapshot) == pytest.approx(30.0)


def test_stage_pool_no_preemption_existing_laxity_uses_bucket_tail() -> None:
    task = _task(deadline_offset_s=10.0)
    key = StagePool._sampling_key_dict(task["sampling_params"])
    pool = StagePool(0, [_FakeDiffusionClient("tcp://host-a:1000/input", None)])
    snapshot = _snapshot(
        safe_admit_capacity=1,
        estimated_step_ms=100.0,
        incremental_step_ms_if_add_one=10.0,
        min_remaining_steps=2,
        max_remaining_steps=9,
        num_running=1,
        num_waiting=1,
        key=key,
        effective_batch_size=1.0,
        min_laxity_ms=100.0,
    )
    snapshot["enable_step_preemption"] = False

    assert pool._estimate_existing_laxity_after_ms(task, snapshot) == pytest.approx(10.0)


def test_stage_pool_token_policy_name_existing_laxity_uses_bucket_tail() -> None:
    task = _task(deadline_offset_s=10.0)
    key = StagePool._sampling_key_dict(task["sampling_params"])
    pool = StagePool(0, [_FakeDiffusionClient("tcp://host-a:1000/input", None)])
    snapshot = _snapshot(
        safe_admit_capacity=1,
        estimated_step_ms=100.0,
        incremental_step_ms_if_add_one=10.0,
        min_remaining_steps=2,
        max_remaining_steps=9,
        num_running=1,
        num_waiting=1,
        key=key,
        effective_batch_size=1.0,
        min_laxity_ms=100.0,
    )
    snapshot["policy"] = "TokenSloStepScheduler"

    assert pool._estimate_existing_laxity_after_ms(task, snapshot) == pytest.approx(10.0)


def test_stage_pool_admit_delay_waits_for_current_step_boundary() -> None:
    task = _task(deadline_offset_s=10.0)
    key = StagePool._sampling_key_dict(task["sampling_params"])
    pool = StagePool(0, [_FakeDiffusionClient("tcp://host-a:1000/input", None)])
    snapshot = _snapshot(
        safe_admit_capacity=1,
        estimated_step_ms=123.0,
        min_remaining_steps=10,
        num_running=1,
        num_waiting=0,
        key=key,
    )

    assert pool._estimate_admit_delay_ms(snapshot, key) == pytest.approx(123.0)


def test_stage_pool_no_preemption_admit_delay_uses_bucket_tail_when_queued() -> None:
    task = _task(deadline_offset_s=10.0)
    key = StagePool._sampling_key_dict(task["sampling_params"])
    pool = StagePool(0, [_FakeDiffusionClient("tcp://host-a:1000/input", None)])
    snapshot = _snapshot(
        safe_admit_capacity=1,
        estimated_step_ms=100.0,
        min_remaining_steps=2,
        max_remaining_steps=9,
        num_running=1,
        num_waiting=1,
        key=key,
    )
    snapshot["enable_step_preemption"] = False

    assert pool._estimate_admit_delay_ms(snapshot, key) == pytest.approx(900.0)


def test_stage_pool_token_policy_name_fallback_uses_no_preemption_tail() -> None:
    task = _task(deadline_offset_s=10.0)
    key = StagePool._sampling_key_dict(task["sampling_params"])
    pool = StagePool(0, [_FakeDiffusionClient("tcp://host-a:1000/input", None)])
    snapshot = _snapshot(
        safe_admit_capacity=1,
        estimated_step_ms=100.0,
        min_remaining_steps=2,
        max_remaining_steps=9,
        num_running=1,
        num_waiting=1,
        key=key,
    )
    snapshot["policy"] = "TokenSloStepScheduler"

    assert pool._estimate_admit_delay_ms(snapshot, key) == pytest.approx(900.0)


def test_stage_pool_preemptive_admit_delay_keeps_same_key_step_boundary_discount() -> None:
    task = _task(deadline_offset_s=10.0)
    key = StagePool._sampling_key_dict(task["sampling_params"])
    pool = StagePool(0, [_FakeDiffusionClient("tcp://host-a:1000/input", None)])
    snapshot = _snapshot(
        safe_admit_capacity=1,
        estimated_step_ms=100.0,
        min_remaining_steps=2,
        max_remaining_steps=9,
        num_running=1,
        num_waiting=1,
        key=key,
    )
    snapshot["enable_step_preemption"] = True

    assert pool._estimate_admit_delay_ms(snapshot, key) == pytest.approx(100.0)
