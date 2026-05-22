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
                "step_cost_model_path": cost_model_path,
                "step_cost_metric": "p90_ms",
            }
        },
    )


def _stage_config_with_missing_model(cost_model_path: str) -> Any:
    return SimpleNamespace(
        model="Missing/Model",
        additional_config={
            "diffusion_slo_scheduler": {
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
