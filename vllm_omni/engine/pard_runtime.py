"""PARD runtime glue for full-DAG control-plane decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from vllm_omni.engine.dag_runtime import DagRuntime
from vllm_omni.engine.dag_types import DagRequestContext
from vllm_omni.engine.pard_broker import PardBrokerDecision, PardRequestBroker, policy_for_alias
from vllm_omni.engine.pard_planner import PardPlannerMode, PardStageLatencyLookup, PardStatePlanner


@dataclass(frozen=True, slots=True)
class PardRuntimeConfig:
    policy_alias: str = "dag_full_pard"
    planner_mode: PardPlannerMode = PardPlannerMode.NORMAL
    lambda_quantile: float = 0.1
    require_full_profiles: bool = True

    def __post_init__(self) -> None:
        alias_mode = planner_mode_for_policy_alias(self.policy_alias)
        if alias_mode is not None:
            object.__setattr__(self, "planner_mode", alias_mode)


def planner_mode_for_policy_alias(alias: str) -> PardPlannerMode | None:
    normalized = alias.strip().lower()
    if normalized == "dag_pard_back":
        return PardPlannerMode.BACK
    if normalized == "dag_pard_sf":
        return PardPlannerMode.SUBSEQUENT_EXEC_ONLY
    if normalized == "dag_pard_lower":
        return PardPlannerMode.LOWER
    if normalized == "dag_pard_upper":
        return PardPlannerMode.UPPER
    return None


class PardRuntime:
    """Owns one PARD Request Broker per DAG stage."""

    def __init__(
        self,
        dag_runtime: DagRuntime,
        latency_lookup: PardStageLatencyLookup,
        *,
        config: PardRuntimeConfig | None = None,
    ) -> None:
        self.dag_runtime = dag_runtime
        self.config = config or PardRuntimeConfig()
        self.planner = PardStatePlanner(
            dag_runtime.config,
            latency_lookup,
            lambda_quantile=self.config.lambda_quantile,
            mode=self.config.planner_mode,
            require_full_profiles=self.config.require_full_profiles,
        )
        policy = policy_for_alias(self.config.policy_alias)
        self.brokers: dict[int, PardRequestBroker] = {}
        for spec in dag_runtime.config.stage_specs:
            profile = latency_lookup.profile_for(spec)
            service_work = 1.0
            if profile is not None and profile.throughput_work_per_s is not None:
                service_work = max(float(profile.throughput_work_per_s), 0.001)
            self.brokers[spec.stage_id] = PardRequestBroker(
                stage_id=spec.stage_id,
                planner=self.planner,
                policy=policy,
                service_work_per_window=service_work,
            )

    def stage_entry(
        self,
        request_id: str,
        stage_id: int,
        *,
        now_s: float | None = None,
        recent_input_work: float = 0.0,
        stage_snapshots: Mapping[int, Mapping[str, Any]] | None = None,
    ) -> PardBrokerDecision:
        ctx = self._get_required_context(request_id)
        return self.brokers[stage_id].stage_entry(
            ctx,
            now_s=now_s,
            recent_input_work=recent_input_work,
            stage_snapshots=stage_snapshots,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy_alias": self.config.policy_alias,
            "planner_mode": self.config.planner_mode.value,
            "lambda_quantile": self.config.lambda_quantile,
            "brokers": {stage_id: broker.snapshot() for stage_id, broker in self.brokers.items()},
        }

    def _get_required_context(self, request_id: str) -> DagRequestContext:
        ctx = self.dag_runtime.get_request_context(request_id)
        if ctx is None:
            raise KeyError(f"Unknown DAG request context: {request_id}")
        return ctx


__all__ = [
    "PardRuntime",
    "PardRuntimeConfig",
    "planner_mode_for_policy_alias",
]
