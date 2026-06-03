from __future__ import annotations

import pytest

from vllm_omni.engine.dag_runtime import build_required_full_dag_template
from vllm_omni.engine.dag_runtime import DagRuntimeConfig
from vllm_omni.engine.dag_types import DagRequestContext, DagStageKind, DagStageResourceSpec, DagStageSpec
from vllm_omni.engine.pard_planner import (
    MissingPardProfilesError,
    MissingPardStagesError,
    PardPlannerMode,
    PardStageLatencyLookup,
    PardStageLatencyProfile,
    PardStatePlanner,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _profiles(*, include_audio: bool = True) -> dict[DagStageKind, PardStageLatencyProfile]:
    profiles = {
        DagStageKind.TEXT_ENCODER: PardStageLatencyProfile(
            DagStageKind.TEXT_ENCODER, 1.0, 2.0, 3.0, throughput_work_per_s=100.0, profile_version="text"
        ),
        DagStageKind.IMAGE_ENCODER: PardStageLatencyProfile(
            DagStageKind.IMAGE_ENCODER, 2.0, 0.0, 4.0, throughput_work_per_s=100.0, profile_version="image"
        ),
        DagStageKind.VAE_ENCODER: PardStageLatencyProfile(
            DagStageKind.VAE_ENCODER, 0.0, 1.0, 5.0, throughput_work_per_s=100.0, profile_version="vae-enc"
        ),
        DagStageKind.DIT_DENOISE: PardStageLatencyProfile(
            DagStageKind.DIT_DENOISE,
            10.0,
            20.0,
            30.0,
            throughput_work_per_s=200.0,
            profile_version="dit",
        ),
        DagStageKind.VAE_DECODER: PardStageLatencyProfile(
            DagStageKind.VAE_DECODER, 3.0, 4.0, 5.0, tail_quantile_ms=1.0, profile_version="vae-dec"
        ),
    }
    if include_audio:
        profiles[DagStageKind.AUDIO_DECODER] = PardStageLatencyProfile(
            DagStageKind.AUDIO_DECODER,
            5.0,
            6.0,
            7.0,
            tail_quantile_ms=2.0,
            profile_version="audio",
        )
    return profiles


def test_planner_estimates_l_pre_l_cur_and_max_downstream_branch() -> None:
    config = build_required_full_dag_template({})
    planner = PardStatePlanner(config, PardStageLatencyLookup(_profiles()))
    runtime_ctx = DagRequestContext(
        request_id="req-1",
        arrival_time_s=100.0,
        deadline_time_s=102.0,
    )

    plan = planner.plan_stage_entry(runtime_ctx, 0, now_s=101.0)

    assert config.spec_by_id[0].kind == DagStageKind.TEXT_ENCODER
    assert plan.l_pre_ms == pytest.approx(1000.0)
    assert plan.l_cur_ms == pytest.approx(6.0)
    assert plan.downstream_path_ms_by_stage[5] == pytest.approx(12.0)
    assert plan.l_sub_ms == pytest.approx(52.0)
    assert plan.estimated_e2e_latency_ms == pytest.approx(1058.0)
    assert plan.remaining_budget_ms == pytest.approx(1000.0)
    assert plan.would_miss_deadline is False


def test_planner_uses_dit_token_cost_estimator() -> None:
    config = build_required_full_dag_template({})

    def _dit_estimator(_ctx, _spec, _snapshot):
        return 123.0, "token_cost_model", {"latent_tokens": 4096}

    planner = PardStatePlanner(
        config,
        PardStageLatencyLookup(_profiles()),
        dit_cost_estimator=_dit_estimator,
    )
    runtime_ctx = DagRequestContext(
        request_id="req-dit",
        arrival_time_s=10.0,
        deadline_time_s=11.0,
    )

    plan = planner.plan_stage_entry(runtime_ctx, 3, now_s=10.1)

    assert plan.current_stage.execution_ms == pytest.approx(123.0)
    assert plan.current_stage.source == "token_cost_model"
    assert plan.current_stage.metadata["latent_tokens"] == 4096


def test_planner_lower_and_upper_ablations_only_change_subsequent_batch_wait() -> None:
    config = build_required_full_dag_template({})
    ctx = DagRequestContext(
        request_id="req-ablation",
        arrival_time_s=1.0,
        deadline_time_s=2.0,
    )

    lower = PardStatePlanner(
        config,
        PardStageLatencyLookup(_profiles()),
        mode=PardPlannerMode.LOWER,
    ).plan_stage_entry(ctx, 3, now_s=1.0)
    upper = PardStatePlanner(
        config,
        PardStageLatencyLookup(_profiles()),
        mode=PardPlannerMode.UPPER,
    ).plan_stage_entry(ctx, 3, now_s=1.0)

    assert lower.current_stage.execution_ms == pytest.approx(30.0)
    assert upper.current_stage.execution_ms == pytest.approx(30.0)
    assert upper.l_sub_ms - lower.l_sub_ms == pytest.approx(7.0)


def test_planner_lambda_quantile_changes_subsequent_batch_wait() -> None:
    config = build_required_full_dag_template({})
    profiles = _profiles()
    profiles[DagStageKind.TEXT_ENCODER] = PardStageLatencyProfile(
        DagStageKind.TEXT_ENCODER,
        1.0,
        2.0,
        3.0,
        tail_wait_quantiles_ms={0.0: 0.0, 0.1: 5.0, 1.0: 20.0},
        profile_version="text-quantiles",
    )
    ctx = DagRequestContext("req-lambda", arrival_time_s=10.0, deadline_time_s=20.0)

    low_lambda = PardStatePlanner(
        config,
        PardStageLatencyLookup(profiles),
        lambda_quantile=0.0,
    ).plan_stage_entry(ctx, 0, now_s=10.0)
    high_lambda = PardStatePlanner(
        config,
        PardStageLatencyLookup(profiles),
        lambda_quantile=1.0,
    ).plan_stage_entry(ctx, 0, now_s=10.0)

    assert high_lambda.l_sub_ms - low_lambda.l_sub_ms == pytest.approx(20.0)


def test_planner_back_and_sf_ablations_change_l_sub_semantics() -> None:
    config = build_required_full_dag_template({})
    ctx = DagRequestContext("req-back-sf", arrival_time_s=10.0, deadline_time_s=20.0)

    back = PardStatePlanner(
        config,
        PardStageLatencyLookup(_profiles()),
        mode=PardPlannerMode.BACK,
    ).plan_stage_entry(ctx, 0, now_s=10.0)
    sf = PardStatePlanner(
        config,
        PardStageLatencyLookup(_profiles()),
        mode=PardPlannerMode.SUBSEQUENT_EXEC_ONLY,
    ).plan_stage_entry(ctx, 0, now_s=10.0)

    assert back.l_sub_ms == pytest.approx(0.0)
    assert sf.l_sub_ms == pytest.approx(37.0)


def test_planner_reports_missing_profiles_without_constants() -> None:
    config = build_required_full_dag_template({})

    with pytest.raises(MissingPardProfilesError) as exc:
        PardStatePlanner(config, PardStageLatencyLookup(_profiles(include_audio=False)))

    assert exc.value.missing == (DagStageKind.AUDIO_DECODER,)

    planner = PardStatePlanner(
        config,
        PardStageLatencyLookup(_profiles(include_audio=False)),
        require_full_profiles=False,
    )
    ctx = DagRequestContext(
        request_id="req-missing",
        arrival_time_s=1.0,
        deadline_time_s=2.0,
    )

    plan = planner.plan_stage_entry(ctx, 4, now_s=1.0)

    assert plan.missing_profiles == (DagStageKind.AUDIO_DECODER,)


def test_planner_rejects_partial_dag_before_full_pard_profiles() -> None:
    resource = DagStageResourceSpec(replica_count=1, cards_per_replica=1)
    config = DagRuntimeConfig(
        stage_specs=(
            DagStageSpec(0, "text_encoder", DagStageKind.TEXT_ENCODER, (), (1,), resource),
            DagStageSpec(1, "dit_denoise", DagStageKind.DIT_DENOISE, (0,), (), resource),
        ),
        entry_stage_ids=(0,),
        exit_stage_ids=(1,),
    )

    with pytest.raises(MissingPardStagesError) as exc:
        PardStatePlanner(config, PardStageLatencyLookup(_profiles()))

    assert DagStageKind.AUDIO_DECODER in exc.value.missing
    assert DagStageKind.VAE_DECODER in exc.value.missing


def test_planner_records_estimate_error_after_observed_stage_completion() -> None:
    config = build_required_full_dag_template({})
    planner = PardStatePlanner(config, PardStageLatencyLookup(_profiles()))
    ctx = DagRequestContext(
        request_id="req-error",
        arrival_time_s=50.0,
        deadline_time_s=52.0,
    )
    ctx.finish_stage(0, now_s=50.5, execution_ms=50.0)

    plan = planner.plan_stage_entry(ctx, 3, now_s=50.5)

    assert plan.estimate_error_ms is not None
