from __future__ import annotations

import pytest

from vllm_omni.engine.dag_cache import DagCacheLocation
from vllm_omni.engine.dag_runtime import DagRuntime, build_required_full_dag_template
from vllm_omni.engine.dag_types import DagStageKind, DagStageLifecycle, DagStageStatus
from vllm_omni.engine.pard_broker import PardBrokerDecisionKind
from vllm_omni.engine.pard_cleanup import PardDropCleanupOutcome, PardDropController, PardExecutionPlacement
from vllm_omni.engine.pard_planner import PardStageLatencyLookup, PardStageLatencyProfile
from vllm_omni.engine.pard_runtime import PardRuntime, PardRuntimeConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeAdapter:
    def __init__(self, stage_id: int) -> None:
        self.stage_id = stage_id
        self.cancel_calls: list[str] = []
        self.cleanup_calls: list[str] = []

    async def cancel(self, request_id: str) -> None:
        self.cancel_calls.append(request_id)

    async def cleanup(self, request_id: str) -> None:
        self.cleanup_calls.append(request_id)


def _profiles() -> dict[DagStageKind, PardStageLatencyProfile]:
    return {
        kind: PardStageLatencyProfile(kind, 1.0, 1.0, 1.0, throughput_work_per_s=100.0)
        for kind in (
            DagStageKind.TEXT_ENCODER,
            DagStageKind.IMAGE_ENCODER,
            DagStageKind.VAE_ENCODER,
            DagStageKind.DIT_DENOISE,
            DagStageKind.VAE_DECODER,
            DagStageKind.AUDIO_DECODER,
        )
    }


def _runtime() -> tuple[DagRuntime, PardRuntime, dict[int, _FakeAdapter]]:
    config = build_required_full_dag_template({})
    adapters = {spec.stage_id: _FakeAdapter(spec.stage_id) for spec in config.stage_specs}
    dag_runtime = DagRuntime(config, adapters)
    pard_runtime = PardRuntime(
        dag_runtime,
        PardStageLatencyLookup(_profiles()),
        config=PardRuntimeConfig(policy_alias="dag_full_pard"),
    )
    return dag_runtime, pard_runtime, adapters


def _drop_decision(pard_runtime: PardRuntime, request_id: str, stage_id: int, *, now_s: float = 100.0):
    return pard_runtime.stage_entry(
        request_id,
        stage_id,
        now_s=now_s,
        recent_input_work=1000.0,
    )


@pytest.mark.asyncio
async def test_pard_cleanup_cancels_queued_request_and_releases_cache() -> None:
    dag_runtime, pard_runtime, adapters = _runtime()
    ctx = dag_runtime.create_request_context("req-queued", arrival_time_s=100.0, deadline_time_s=100.001)
    handle = dag_runtime.create_cache_ref(
        "req-queued",
        producer_stage_id=0,
        consumer_stage_ids={3},
        size_bytes=64,
        location=DagCacheLocation(stage_id=0, replica_id=0, device="npu:0"),
    )
    ctx.stage_lifecycle[0] = DagStageLifecycle(
        stage_id=0,
        status=DagStageStatus.WAITING,
    )

    decision = _drop_decision(pard_runtime, "req-queued", 0)
    result = await pard_runtime.apply_drop_decision(decision)

    assert decision.kind == PardBrokerDecisionKind.PROACTIVE_DROP
    assert result.outcome == PardDropCleanupOutcome.CLEANED_UP
    assert result.placement == PardExecutionPlacement.QUEUED
    assert adapters[0].cancel_calls == ["req-queued"]
    assert all(adapter.cleanup_calls == ["req-queued"] for adapter in adapters.values())
    assert dag_runtime.cache_registry.get(handle).released is True
    assert ctx.trace[-1].event == "pard_drop_cleanup"


@pytest.mark.asyncio
async def test_pard_cleanup_cancels_batched_request() -> None:
    dag_runtime, pard_runtime, adapters = _runtime()
    ctx = dag_runtime.create_request_context("req-batched", arrival_time_s=100.0, deadline_time_s=100.001)
    ctx.enter_stage(0, replica_id=0, now_s=100.0)
    ctx.stage_lifecycle[0].metadata["batch_admitted"] = True

    decision = _drop_decision(pard_runtime, "req-batched", 0)
    result = await pard_runtime.apply_drop_decision(decision)

    assert result.placement == PardExecutionPlacement.BATCHED
    assert result.outcome == PardDropCleanupOutcome.CLEANED_UP
    assert adapters[0].cancel_calls == ["req-batched"]


@pytest.mark.asyncio
async def test_pard_cleanup_defers_dit_running_step_until_step_boundary() -> None:
    dag_runtime, pard_runtime, adapters = _runtime()
    ctx = dag_runtime.create_request_context("req-running", arrival_time_s=100.0, deadline_time_s=100.001)
    ctx.enter_stage(3, replica_id=0, now_s=100.0)
    ctx.stage_lifecycle[3].metadata["inside_denoise_step"] = True

    decision = _drop_decision(pard_runtime, "req-running", 3)
    pending = await pard_runtime.apply_drop_decision(decision)

    assert pending.outcome == PardDropCleanupOutcome.PENDING_STEP_BOUNDARY
    assert pending.pending_step_boundary is True
    assert adapters[3].cancel_calls == []
    assert ctx.stage_lifecycle[3].metadata["pard_drop_pending_step_boundary"] is True

    applied = await pard_runtime.apply_pending_step_boundary("req-running", 3)

    assert applied.outcome == PardDropCleanupOutcome.CLEANED_UP
    assert applied.placement == PardExecutionPlacement.RUNNING_STEP
    assert adapters[3].cancel_calls == ["req-running"]
    assert ctx.stage_lifecycle[3].metadata["pard_step_boundary_drop_applied"] is True


@pytest.mark.asyncio
async def test_pard_cleanup_noops_completed_request_without_resource_cancel() -> None:
    dag_runtime, pard_runtime, adapters = _runtime()
    ctx = dag_runtime.create_request_context("req-done", arrival_time_s=100.0, deadline_time_s=100.001)
    ctx.finish_stage(0, now_s=100.0, execution_ms=1.0)

    decision = _drop_decision(pard_runtime, "req-done", 0)
    result = await pard_runtime.apply_drop_decision(decision)

    assert result.outcome == PardDropCleanupOutcome.COMPLETED_NOOP
    assert result.placement == PardExecutionPlacement.COMPLETED
    assert all(not adapter.cancel_calls for adapter in adapters.values())
    assert all(not adapter.cleanup_calls for adapter in adapters.values())


@pytest.mark.asyncio
async def test_pard_cleanup_is_idempotent_after_cleanup_applied() -> None:
    _dag_runtime, pard_runtime, adapters = _runtime()
    ctx = pard_runtime.dag_runtime.create_request_context("req-once", arrival_time_s=100.0, deadline_time_s=100.001)
    ctx.stage_lifecycle[0] = DagStageLifecycle(
        stage_id=0,
        status=DagStageStatus.WAITING,
    )
    decision = _drop_decision(pard_runtime, "req-once", 0)

    first = await pard_runtime.apply_drop_decision(decision)
    second = await pard_runtime.apply_drop_decision(decision)

    assert first.outcome == PardDropCleanupOutcome.CLEANED_UP
    assert second.outcome == PardDropCleanupOutcome.ALREADY_CLEANED
    assert adapters[0].cancel_calls == ["req-once"]
