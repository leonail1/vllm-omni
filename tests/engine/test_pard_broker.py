from __future__ import annotations

from dataclasses import replace

import pytest

from vllm_omni.engine.dag_runtime import build_required_full_dag_template
from vllm_omni.engine.dag_types import DagRequestContext, DagStageKind
from vllm_omni.engine.pard_broker import (
    PardBrokerDecisionKind,
    PardBrokerPolicy,
    PardRequestBroker,
    policy_for_alias,
)
from vllm_omni.engine.pard_planner import (
    PardPlannerMode,
    PardStageLatencyLookup,
    PardStageLatencyProfile,
    PardStatePlanner,
)
from vllm_omni.engine.pard_priority import PardPriorityPolicy
from vllm_omni.engine.pard_runtime import PardRuntime, PardRuntimeConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _profiles() -> dict[DagStageKind, PardStageLatencyProfile]:
    return {
        DagStageKind.TEXT_ENCODER: PardStageLatencyProfile(
            DagStageKind.TEXT_ENCODER, 1.0, 1.0, 1.0, throughput_work_per_s=100.0
        ),
        DagStageKind.IMAGE_ENCODER: PardStageLatencyProfile(
            DagStageKind.IMAGE_ENCODER, 1.0, 1.0, 1.0, throughput_work_per_s=100.0
        ),
        DagStageKind.VAE_ENCODER: PardStageLatencyProfile(
            DagStageKind.VAE_ENCODER, 1.0, 1.0, 1.0, throughput_work_per_s=100.0
        ),
        DagStageKind.DIT_DENOISE: PardStageLatencyProfile(
            DagStageKind.DIT_DENOISE, 10.0, 10.0, 10.0, throughput_work_per_s=50.0
        ),
        DagStageKind.VAE_DECODER: PardStageLatencyProfile(
            DagStageKind.VAE_DECODER, 1.0, 1.0, 1.0, throughput_work_per_s=100.0
        ),
        DagStageKind.AUDIO_DECODER: PardStageLatencyProfile(
            DagStageKind.AUDIO_DECODER, 1.0, 1.0, 1.0, throughput_work_per_s=100.0
        ),
    }


def _broker(policy: PardBrokerPolicy | None = None) -> PardRequestBroker:
    config = build_required_full_dag_template({})
    planner = PardStatePlanner(config, PardStageLatencyLookup(_profiles()))
    return PardRequestBroker(
        stage_id=0,
        planner=planner,
        policy=policy or PardBrokerPolicy(priority_policy=PardPriorityPolicy.ADAPTIVE),
        service_work_per_window=100.0,
    )


def test_broker_accepts_request_and_enqueues_with_trace() -> None:
    broker = _broker()
    ctx = DagRequestContext("req-ok", arrival_time_s=100.0, deadline_time_s=101.0)

    decision = broker.stage_entry(ctx, now_s=100.0, recent_input_work=20.0)

    assert decision.kind == PardBrokerDecisionKind.ACCEPT
    assert decision.accepted is True
    assert len(broker.queue) == 1
    assert ctx.dropped is False
    assert ctx.trace[-1].event == "pard_broker_decision"
    assert ctx.trace[-1].details["kind"] == "accept"
    assert ctx.trace[-1].details["policy_name"] == "dag_full_pard"
    assert ctx.trace[-1].details["priority_policy"] == "adaptive"
    assert ctx.trace[-1].details["recent_input_work"] == pytest.approx(20.0)
    assert ctx.trace[-1].details["stage_service_work"] == pytest.approx(100.0)
    assert ctx.trace[-1].details["smoothed_input_work"] == pytest.approx(20.0)
    assert ctx.trace[-1].details["current_stage_profile_version"] == "unknown"


def test_broker_proactively_drops_predicted_slo_miss() -> None:
    broker = _broker()
    ctx = DagRequestContext("req-drop", arrival_time_s=100.0, deadline_time_s=100.01)
    ctx.finish_stage(1, now_s=100.001, execution_ms=5.0)

    decision = broker.stage_entry(ctx, now_s=100.0, recent_input_work=200.0)

    assert decision.kind == PardBrokerDecisionKind.PROACTIVE_DROP
    assert decision.dropped is True
    assert decision.reason == "predicted_e2e_slo_miss"
    assert decision.invalid_compute_ms == pytest.approx(5.0)
    assert ctx.dropped is True
    assert ctx.drop_stage_id == 0
    assert ctx.trace[-1].details["invalid_compute_ms"] == pytest.approx(5.0)
    assert len(broker.queue) == 0


def test_broker_reactively_drops_already_late_request() -> None:
    broker = _broker(PardBrokerPolicy(proactive_drop=False, reactive_drop=True))
    ctx = DagRequestContext("req-late", arrival_time_s=100.0, deadline_time_s=100.5)

    decision = broker.stage_entry(ctx, now_s=101.0, recent_input_work=20.0)

    assert decision.kind == PardBrokerDecisionKind.REACTIVE_DROP
    assert decision.reason == "deadline_already_missed"
    assert ctx.dropped is True


def test_no_drop_policy_accepts_even_when_predicted_late() -> None:
    broker = _broker(policy_for_alias("dag_no_drop"))
    ctx = DagRequestContext("req-no-drop", arrival_time_s=100.0, deadline_time_s=100.01)

    decision = broker.stage_entry(ctx, now_s=100.0, recent_input_work=200.0)

    assert decision.kind == PardBrokerDecisionKind.ACCEPT
    assert ctx.dropped is False
    assert len(broker.queue) == 1


def test_hbf_broker_pops_high_budget_request_first() -> None:
    broker = _broker(policy_for_alias("dag_pard_hbf"))
    high = DagRequestContext("req-high", arrival_time_s=100.0, deadline_time_s=102.0)
    low = DagRequestContext("req-low", arrival_time_s=100.0, deadline_time_s=101.0)
    broker.stage_entry(low, now_s=100.0, recent_input_work=20.0)
    broker.stage_entry(high, now_s=100.0, recent_input_work=20.0)

    next_item = broker.pop_next(recent_input_work=20.0)

    assert next_item.request_id == "req-high"


def test_paper_native_aliases_have_distinct_broker_semantics() -> None:
    assert policy_for_alias("dag_pard_back").ablation_semantic == "ignore_subsequent_modules_lsub_zero"
    assert policy_for_alias("dag_pard_sf").ablation_semantic == "subsequent_execution_only"
    assert policy_for_alias("dag_pard_oc").overload_control is True
    assert policy_for_alias("dag_pard_split").stage_budget_mode == "fixed_split"
    assert policy_for_alias("dag_pard_wcl").stage_budget_mode == "wcl"
    assert policy_for_alias("dag_pard_fcfs").priority_policy == PardPriorityPolicy.FCFS
    assert policy_for_alias("dag_pard_hbf").priority_policy == PardPriorityPolicy.HBF
    assert policy_for_alias("dag_pard_lbf").priority_policy == PardPriorityPolicy.LBF


def test_overload_control_alias_drops_when_queue_delay_exceeds_threshold() -> None:
    config = build_required_full_dag_template({})
    planner = PardStatePlanner(config, PardStageLatencyLookup(_profiles()))
    broker = PardRequestBroker(
        stage_id=0,
        planner=planner,
        policy=replace(policy_for_alias("dag_pard_oc"), overload_admit_fraction=0.0),
        service_work_per_window=100.0,
    )
    ctx = DagRequestContext("req-oc", arrival_time_s=100.0, deadline_time_s=101.0)

    decision = broker.stage_entry(
        ctx,
        now_s=100.0,
        recent_input_work=20.0,
        stage_snapshots={0: {"queue_wait_ms": 200.0}},
    )

    assert decision.kind == PardBrokerDecisionKind.PROACTIVE_DROP
    assert decision.reason == "overload_control_queue_delay"


def test_overload_control_uses_attempted_arrivals_for_admit_rate() -> None:
    config = build_required_full_dag_template({})
    planner = PardStatePlanner(config, PardStageLatencyLookup(_profiles()))
    broker = PardRequestBroker(
        stage_id=0,
        planner=planner,
        policy=policy_for_alias("dag_pard_oc"),
        service_work_per_window=100.0,
    )
    snapshots = {0: {"queue_wait_ms": 200.0}}

    first = broker.stage_entry(
        DagRequestContext("req-oc-1", arrival_time_s=100.0, deadline_time_s=101.0),
        now_s=100.0,
        recent_input_work=20.0,
        stage_snapshots=snapshots,
    )
    second = broker.stage_entry(
        DagRequestContext("req-oc-2", arrival_time_s=100.0, deadline_time_s=101.0),
        now_s=100.0,
        recent_input_work=20.0,
        stage_snapshots=snapshots,
    )

    assert first.kind == PardBrokerDecisionKind.PROACTIVE_DROP
    assert second.kind == PardBrokerDecisionKind.ACCEPT


def test_split_alias_uses_per_module_budget() -> None:
    broker = _broker(policy_for_alias("dag_pard_split"))
    ctx = DagRequestContext("req-split", arrival_time_s=100.0, deadline_time_s=100.012)

    decision = broker.stage_entry(ctx, now_s=100.0, recent_input_work=20.0)

    assert decision.kind == PardBrokerDecisionKind.PROACTIVE_DROP
    assert decision.reason == "fixed_split_stage_budget_exceeded"


def test_already_dropped_request_is_not_reaccepted_downstream() -> None:
    broker = _broker()
    ctx = DagRequestContext("req-already-drop", arrival_time_s=100.0, deadline_time_s=101.0)
    ctx.mark_dropped(0, "upstream_drop", now_s=100.0)

    decision = broker.stage_entry(ctx, now_s=100.1, recent_input_work=20.0)

    assert decision.kind == PardBrokerDecisionKind.REACTIVE_DROP
    assert decision.reason == "request_already_dropped"
    assert len(broker.queue) == 0
    assert ctx.drop_stage_id == 0
    assert ctx.drop_reason == "upstream_drop"
    assert ctx.trace[-1].details["original_drop_stage_id"] == 0
    assert ctx.trace[-1].details["original_drop_reason"] == "upstream_drop"


def test_pard_runtime_builds_one_broker_per_stage() -> None:
    from vllm_omni.engine.dag_runtime import DagRuntime

    dag_runtime = DagRuntime(build_required_full_dag_template({}))
    dag_runtime.create_request_context("req-runtime", arrival_time_s=100.0, deadline_time_s=101.0)
    runtime = PardRuntime(
        dag_runtime,
        PardStageLatencyLookup(_profiles()),
        config=PardRuntimeConfig(policy_alias="dag_full_pard"),
    )

    decision = runtime.stage_entry("req-runtime", 0, now_s=100.0, recent_input_work=20.0)

    assert decision.kind == PardBrokerDecisionKind.ACCEPT
    assert len(runtime.brokers) == 6
    assert runtime.snapshot()["policy_alias"] == "dag_full_pard"


@pytest.mark.parametrize(
    ("alias", "expected_mode"),
    [
        ("dag_pard_back", PardPlannerMode.BACK),
        ("dag_pard_sf", PardPlannerMode.SUBSEQUENT_EXEC_ONLY),
        ("dag_pard_lower", PardPlannerMode.LOWER),
        ("dag_pard_upper", PardPlannerMode.UPPER),
    ],
)
def test_runtime_alias_selects_required_planner_mode(alias: str, expected_mode: PardPlannerMode) -> None:
    config = PardRuntimeConfig(policy_alias=alias)

    assert config.planner_mode == expected_mode
