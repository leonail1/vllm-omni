from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

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
    MissingPardProfilesError,
    MissingPardStagesError,
    PardPlannerMode,
    PardStageLatencyLookup,
    PardStageLatencyProfile,
    PardStatePlanner,
)
from vllm_omni.engine.pard_priority import PardPriorityPolicy
from vllm_omni.engine.pard_runtime import (
    PardRuntime,
    PardRuntimeConfig,
    build_pard_runtime_from_stage_pools,
)

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


class _PardConfigPool:
    def __init__(
        self,
        stage_id: int,
        *,
        stage_type: str = "llm",
        model_stage: str,
        final_output: bool = False,
        final_output_type: str | None = None,
        engine_input_source: list[int] | None = None,
        additional_config: dict | None = None,
    ) -> None:
        self.stage_id = stage_id
        self._stage_type = stage_type
        self.clients = [
            SimpleNamespace(
                model_stage=model_stage,
                final_output=final_output,
                final_output_type=final_output_type,
                engine_input_source=list(engine_input_source or []),
                devices="",
            )
        ]
        self.stage_vllm_config = SimpleNamespace(max_num_seqs=4, additional_config=additional_config or {})
        self.stage_slo_config = self.stage_vllm_config

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
        return [0]

    def dag_replica_device_ids(self, _replica_id: int):
        return ()

    def dag_replica_card_count(self, _replica_id: int):
        return 1

    async def abort_requests(self, _request_ids):
        return None

    def release_bindings(self, _request_ids):
        return None


def _profile_config(profile: PardStageLatencyProfile) -> dict[str, object]:
    return {
        "queue_wait_ms": profile.queue_wait_ms,
        "batch_wait_ms": profile.batch_wait_ms,
        "execution_ms": profile.execution_ms,
        "tail_quantile_ms": profile.tail_quantile_ms,
        "tail_wait_quantiles_ms": {"0.1": 1.5, "0.9": 4.0},
        "throughput_work_per_s": profile.throughput_work_per_s,
        "profile_version": "unit-test",
    }


def _pard_runtime_config(*, include_profiles: bool = True) -> dict[str, object]:
    raw: dict[str, object] = {
        "enabled": True,
        "policy_alias": "dag_pard_lbf",
        "lambda_quantile": 0.2,
        "completed_trace_limit": 3,
    }
    if include_profiles:
        raw["profiles"] = {kind.value: _profile_config(profile) for kind, profile in _profiles().items()}
    return {"dag_pard_runtime": raw}


def _full_pard_stage_pools(*, additional_config: dict | None = None) -> list[_PardConfigPool]:
    return [
        _PardConfigPool(0, model_stage="text_encoder", engine_input_source=[], additional_config=additional_config),
        _PardConfigPool(1, model_stage="image_encoder", engine_input_source=[]),
        _PardConfigPool(2, model_stage="vae_encoder", engine_input_source=[1]),
        _PardConfigPool(3, stage_type="diffusion", model_stage="diffusion", engine_input_source=[0, 2]),
        _PardConfigPool(4, model_stage="vae_decoder", final_output=True, engine_input_source=[3]),
        _PardConfigPool(
            5,
            model_stage="audio_decoder",
            final_output=True,
            final_output_type="audio",
            engine_input_source=[3],
        ),
    ]


def _broker(policy: PardBrokerPolicy | None = None) -> PardRequestBroker:
    config = build_required_full_dag_template({})
    planner = PardStatePlanner(config, PardStageLatencyLookup(_profiles()))
    return PardRequestBroker(
        stage_id=0,
        planner=planner,
        policy=policy or PardBrokerPolicy(priority_policy=PardPriorityPolicy.ADAPTIVE),
        service_work_per_window=100.0,
    )


def test_pard_runtime_builder_is_disabled_without_explicit_config() -> None:
    assert build_pard_runtime_from_stage_pools(_full_pard_stage_pools()) is None


def test_pard_runtime_builder_uses_explicit_full_dag_config() -> None:
    runtime = build_pard_runtime_from_stage_pools(
        _full_pard_stage_pools(additional_config=_pard_runtime_config()),
    )

    assert runtime is not None
    assert runtime.config.policy_alias == "dag_pard_lbf"
    assert runtime.config.lambda_quantile == pytest.approx(0.2)
    assert runtime.dag_runtime.completed_trace_limit == 3
    assert set(runtime.brokers) == {0, 1, 2, 3, 4, 5}
    assert {spec.kind for spec in runtime.dag_runtime.config.stage_specs} == set(_profiles())


def test_pard_runtime_builder_rejects_missing_enabled_profiles() -> None:
    with pytest.raises(MissingPardProfilesError):
        build_pard_runtime_from_stage_pools(
            _full_pard_stage_pools(additional_config=_pard_runtime_config(include_profiles=False)),
        )


def test_pard_runtime_builder_rejects_configured_profile_gate_bypass() -> None:
    config = _pard_runtime_config(include_profiles=False)
    config["dag_pard_runtime"]["require_full_profiles"] = False

    with pytest.raises(ValueError, match="requires complete full-DAG stages and profiles"):
        build_pard_runtime_from_stage_pools(_full_pard_stage_pools(additional_config=config))


def test_pard_runtime_builder_rejects_partial_dag_when_enabled() -> None:
    with pytest.raises(MissingPardStagesError):
        build_pard_runtime_from_stage_pools(
            [
                _PardConfigPool(
                    0,
                    model_stage="text_encoder",
                    engine_input_source=[],
                    additional_config=_pard_runtime_config(),
                )
            ]
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
