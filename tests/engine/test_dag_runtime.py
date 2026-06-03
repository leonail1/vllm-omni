from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.engine.dag_cache import DagCacheLocation
from vllm_omni.engine.dag_runtime import (
    DagRuntime,
    DagRuntimeConfig,
    build_dag_config_from_stage_pools,
    build_required_full_dag_template,
)
from vllm_omni.engine.dag_types import (
    FULL_DAG_REQUIRED_STAGE_KINDS,
    DagCacheMovementKind,
    DagStageKind,
    DagStageResourceSpec,
    DagStageSpec,
)
from vllm_omni.engine.stage_adapters import build_stage_spec_from_pool
from vllm_omni.engine.stage_adapters import StagePoolDagAdapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakePool:
    def __init__(
        self,
        stage_id: int,
        *,
        stage_type: str = "llm",
        model_stage: str | None = None,
        final_output: bool = False,
        final_output_type: str | None = None,
        num_replicas: int = 1,
        devices: list[str] | None = None,
        additional_config: dict | None = None,
        engine_input_source: list[int] | None = None,
        replica_card_counts: list[int] | None = None,
    ) -> None:
        self.stage_id = stage_id
        self._stage_type = stage_type
        self.clients = [
            SimpleNamespace(
                model_stage=model_stage,
                final_output=final_output,
                final_output_type=final_output_type,
                devices=(devices or [""])[idx] if devices else "",
                engine_input_source=list(engine_input_source or []),
            )
            for idx in range(num_replicas)
        ]
        self.stage_vllm_config = SimpleNamespace(max_num_seqs=4, additional_config=additional_config or {})
        self.stage_slo_config = self.stage_vllm_config
        self._replica_card_counts = list(replica_card_counts or [])

    @property
    def stage_client(self):
        return self.clients[0]

    @property
    def stage_type(self) -> str:
        return self._stage_type

    @property
    def final_output(self) -> bool:
        return bool(self.stage_client.final_output)

    def live_replica_ids(self):
        return list(range(len(self.clients)))

    def dag_replica_device_ids(self, replica_id: int):
        device_attr = self.clients[replica_id].devices
        return tuple(part.strip() for part in device_attr.split(",") if part.strip())

    def dag_replica_card_count(self, replica_id: int):
        if replica_id < len(self._replica_card_counts):
            return self._replica_card_counts[replica_id]
        return max(1, len(self.dag_replica_device_ids(replica_id)))

    async def abort_requests(self, _request_ids):
        return None

    def release_bindings(self, _request_ids):
        return None


def test_required_full_dag_template_covers_all_roles_and_branch_merge() -> None:
    config = build_required_full_dag_template({})

    assert {spec.kind for spec in config.stage_specs} == FULL_DAG_REQUIRED_STAGE_KINDS
    assert config.entry_stage_ids == (0, 1)
    assert config.exit_stage_ids == (4, 5)

    spec_by_id = config.spec_by_id
    assert spec_by_id[3].kind == DagStageKind.DIT_DENOISE
    assert spec_by_id[3].pres == (0, 2)
    assert spec_by_id[3].subs == (4, 5)


def test_runtime_config_rejects_cycles_and_missing_reverse_edges() -> None:
    resource = DagStageResourceSpec(replica_count=1, cards_per_replica=1)
    with pytest.raises(ValueError, match="reverse edge"):
        DagRuntimeConfig(
            stage_specs=(
                DagStageSpec(0, "a", DagStageKind.TEXT_ENCODER, (), (1,), resource),
                DagStageSpec(1, "b", DagStageKind.DIT_DENOISE, (), (), resource),
            ),
            entry_stage_ids=(0, 1),
            exit_stage_ids=(1,),
        )

    with pytest.raises(ValueError, match="acyclic"):
        DagRuntimeConfig(
            stage_specs=(
                DagStageSpec(0, "a", DagStageKind.TEXT_ENCODER, (1,), (1,), resource),
                DagStageSpec(1, "b", DagStageKind.DIT_DENOISE, (0,), (0,), resource),
            ),
            entry_stage_ids=(),
            exit_stage_ids=(),
        )


def test_stage_pool_spec_infers_explicit_kind_and_resource_layout() -> None:
    pool = _FakePool(
        2,
        stage_type="diffusion",
        model_stage="diffusion",
        num_replicas=2,
        devices=["", ""],
        replica_card_counts=[2, 2],
        additional_config={"dag_stage_kind": "dit_denoise", "diffusion_scheduler_policy": "token_objective"},
    )

    spec = build_stage_spec_from_pool(pool, pres=(0, 1), subs=(3,))

    assert spec.kind == DagStageKind.DIT_DENOISE
    assert spec.pres == (0, 1)
    assert spec.subs == (3,)
    assert spec.resource.replica_count == 2
    assert spec.resource.cards_per_replica == 2
    assert spec.batch.allow_step_boundary_preemption is True
    assert spec.batch.scheduler_policy == "token_objective"


def test_runtime_from_stage_pools_preserves_branch_merge_edges() -> None:
    pools = [
        _FakePool(0, stage_type="llm", model_stage="text_encoder"),
        _FakePool(1, stage_type="llm", model_stage="image_encoder"),
        _FakePool(2, stage_type="llm", model_stage="vae_encoder", engine_input_source=[1]),
        _FakePool(3, stage_type="diffusion", model_stage="diffusion", engine_input_source=[0, 2]),
        _FakePool(4, stage_type="llm", model_stage="vae_decoder", final_output=True, engine_input_source=[3]),
        _FakePool(
            5,
            stage_type="llm",
            model_stage="audio_decoder",
            final_output=True,
            final_output_type="audio",
            engine_input_source=[3],
        ),
    ]

    config = build_dag_config_from_stage_pools(pools)
    spec_by_id = config.spec_by_id

    assert config.entry_stage_ids == (0, 1)
    assert config.exit_stage_ids == (4, 5)
    assert spec_by_id[3].pres == (0, 2)
    assert spec_by_id[3].subs == (4, 5)
    assert {spec.kind for spec in config.stage_specs} == FULL_DAG_REQUIRED_STAGE_KINDS


def test_runtime_from_stage_pools_ignores_self_input_source() -> None:
    pools = [_FakePool(0, stage_type="llm", model_stage="text_encoder", engine_input_source=[0])]

    config = build_dag_config_from_stage_pools(pools)

    assert config.entry_stage_ids == (0,)
    assert config.exit_stage_ids == (0,)
    assert config.spec_by_id[0].pres == ()
    assert config.spec_by_id[0].subs == ()


def test_dag_runtime_tracks_request_lifecycle_and_cache_trace() -> None:
    config = build_required_full_dag_template({})
    runtime = DagRuntime(config)

    ctx = runtime.create_request_context(
        "req-1",
        arrival_time_s=100.0,
        deadline_time_s=105.0,
        metadata={"source": "unit-test"},
    )
    runtime.enter_stage("req-1", 0, replica_id=1)
    runtime.finish_stage("req-1", 0, execution_ms=3.5, metadata={"tokens": 16})
    handle_id = runtime.create_cache_ref(
        "req-1",
        producer_stage_id=0,
        consumer_stage_ids={3},
        size_bytes=32,
        location=DagCacheLocation(stage_id=0, replica_id=1, device="npu:1"),
    )
    runtime.cache_registry.mark_ready(handle_id)
    runtime.cache_registry.prefetch(handle_id, consumer_stage_id=3)
    runtime.cache_registry.pull(handle_id, consumer_stage_id=3, start_s=100.0)

    assert ctx.remaining_budget_ms(now_s=104.5) == pytest.approx(500.0)
    assert ctx.completed_stage_ids == {0}
    assert handle_id in ctx.cache_refs
    assert [event.movement for event in runtime.cache_registry.trace] == [
        DagCacheMovementKind.PUSH,
        DagCacheMovementKind.PREFETCH,
        DagCacheMovementKind.PULL,
    ]

    runtime.cleanup_request("req-1")
    assert runtime.get_request_context("req-1") is None
    assert runtime.get_completed_trace("req-1") is None
    assert runtime.cache_registry.get(handle_id).released is True


def test_dag_runtime_archives_completed_traces_only_when_explicitly_requested() -> None:
    runtime = DagRuntime(build_required_full_dag_template({}), completed_trace_limit=1)

    runtime.create_request_context("req-1", arrival_time_s=100.0)
    runtime.enter_stage("req-1", 0)
    runtime.archive_request_trace("req-1")
    runtime.cleanup_request("req-1")

    assert runtime.get_completed_trace("req-1") is not None

    runtime.create_request_context("req-2", arrival_time_s=101.0)
    runtime.enter_stage("req-2", 0)
    runtime.archive_request_trace("req-2")
    runtime.cleanup_request("req-2")

    assert runtime.get_completed_trace("req-1") is None
    assert runtime.get_completed_trace("req-2") is not None


def test_runtime_from_stage_pools_exposes_missing_required_roles() -> None:
    runtime = DagRuntime.from_stage_pools(
        [
            _FakePool(0, stage_type="llm", model_stage="text_encoder"),
            _FakePool(1, stage_type="diffusion", model_stage="diffusion", final_output=True, final_output_type="image"),
        ]
    )

    missing = runtime.validate_required_full_dag_roles()

    assert DagStageKind.TEXT_ENCODER not in missing
    assert DagStageKind.DIT_DENOISE not in missing
    assert DagStageKind.AUDIO_DECODER in missing
    assert runtime.snapshot()["stages"][1]["kind"] == "dit_denoise"


def test_stage_pool_dag_adapter_polls_llm_outputs() -> None:
    class _PollPool(_FakePool):
        def __init__(self) -> None:
            super().__init__(0, stage_type="llm", model_stage="text_encoder")

        async def poll_llm_raw_output(self, replica_id: int, *, timeout_s: float = 0.001):
            return SimpleNamespace(outputs=[object()], timestamp=10.0, scheduler_stats=None)

        async def process_llm_raw_outputs(self, replica_id: int, raw_outputs):
            return [SimpleNamespace(request_id="req-llm", finished=True)]

    pool = _PollPool()
    spec = build_stage_spec_from_pool(pool, pres=(), subs=())
    adapter = StagePoolDagAdapter(pool, spec)

    import asyncio

    outputs = asyncio.run(adapter.poll())

    assert len(outputs) == 1
    assert outputs[0].request_id == "req-llm"
    assert outputs[0].finished is True
