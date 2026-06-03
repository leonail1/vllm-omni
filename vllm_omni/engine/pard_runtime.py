"""PARD runtime glue for full-DAG control-plane decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from vllm_omni.engine.dag_runtime import DagRuntime
from vllm_omni.engine.dag_types import DagRequestContext, DagStageKind
from vllm_omni.engine.pard_broker import PardBrokerDecision, PardRequestBroker, policy_for_alias
from vllm_omni.engine.pard_cleanup import PardDropCleanupResult, PardDropController
from vllm_omni.engine.pard_planner import (
    PardPlannerMode,
    PardStageLatencyLookup,
    PardStageLatencyProfile,
    PardStatePlanner,
)


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


def build_pard_runtime_from_stage_pools(stage_pools: list[Any]) -> PardRuntime | None:
    """Build a PARD runtime from explicit per-stage additional_config.

    The production default is intentionally ``None``. Full-DAG PARD is enabled
    only when a stage config contains ``dag_pard_runtime`` or ``pard_runtime``
    with ``enabled: true``. Missing full-DAG stages/profiles then remain hard
    errors through ``PardStatePlanner``.
    """

    raw_config = _collect_pard_runtime_config(stage_pools)
    if raw_config is None or not _as_bool(raw_config.get("enabled"), default=False):
        return None
    completed_trace_limit = _as_int(raw_config.get("completed_trace_limit"), default=256)
    dag_runtime = DagRuntime.from_stage_pools(stage_pools, completed_trace_limit=completed_trace_limit)
    if "require_full_profiles" in raw_config and not _as_bool(raw_config.get("require_full_profiles"), default=True):
        raise ValueError("Configured full-DAG PARD requires complete full-DAG stages and profiles")
    runtime_config = PardRuntimeConfig(
        policy_alias=str(raw_config.get("policy_alias") or raw_config.get("policy") or "dag_full_pard"),
        lambda_quantile=_as_float(raw_config.get("lambda_quantile"), default=0.1),
        require_full_profiles=True,
    )
    latency_lookup = PardStageLatencyLookup(
        _parse_profiles_by_kind(raw_config.get("profiles") or raw_config.get("profiles_by_kind") or {}),
        profiles_by_stage_id=_parse_profiles_by_stage_id(raw_config.get("profiles_by_stage_id") or {}),
    )
    return PardRuntime(dag_runtime, latency_lookup, config=runtime_config)


def _collect_pard_runtime_config(stage_pools: list[Any]) -> dict[str, Any] | None:
    collected: dict[str, Any] | None = None
    for pool in stage_pools:
        for stage_cfg_attr in ("stage_vllm_config", "stage_slo_config"):
            stage_cfg = getattr(pool, stage_cfg_attr, None)
            additional_config = _as_mapping(getattr(stage_cfg, "additional_config", None))
            if additional_config is None:
                continue
            raw = (
                additional_config.get("dag_pard_runtime")
                or additional_config.get("pard_runtime")
                or additional_config.get("full_dag_pard")
            )
            if raw is None:
                continue
            candidate = {"enabled": raw} if isinstance(raw, bool) else _as_mapping(raw)
            if candidate is None:
                raise TypeError("dag_pard_runtime/pard_runtime must be a boolean or mapping")
            collected = _deep_merge(collected or {}, candidate)
    return collected


def _parse_profiles_by_kind(raw: Any) -> dict[DagStageKind, PardStageLatencyProfile]:
    mapping = _as_mapping(raw)
    if mapping is None:
        raise TypeError("PARD profiles must be a mapping")
    profiles: dict[DagStageKind, PardStageLatencyProfile] = {}
    for kind_name, profile_raw in mapping.items():
        kind = _parse_stage_kind(kind_name)
        profiles[kind] = _parse_profile(kind, profile_raw)
    return profiles


def _parse_profiles_by_stage_id(raw: Any) -> dict[int, PardStageLatencyProfile]:
    mapping = _as_mapping(raw)
    if mapping is None:
        raise TypeError("PARD profiles_by_stage_id must be a mapping")
    profiles: dict[int, PardStageLatencyProfile] = {}
    for stage_id_raw, profile_raw in mapping.items():
        profile_mapping = _required_mapping(profile_raw, "PARD stage profile")
        kind_raw = profile_mapping.get("stage_kind") or profile_mapping.get("kind")
        if kind_raw is None:
            raise ValueError(f"PARD profile for stage id {stage_id_raw!r} requires stage_kind")
        profiles[int(stage_id_raw)] = _parse_profile(_parse_stage_kind(kind_raw), profile_mapping)
    return profiles


def _parse_profile(kind: DagStageKind, raw: Any) -> PardStageLatencyProfile:
    mapping = _required_mapping(raw, f"PARD profile for {kind.value}")
    return PardStageLatencyProfile(
        kind,
        queue_wait_ms=_required_float(mapping, "queue_wait_ms"),
        batch_wait_ms=_required_float(mapping, "batch_wait_ms"),
        execution_ms=_required_float(mapping, "execution_ms"),
        tail_quantile_ms=_as_float(mapping.get("tail_quantile_ms"), default=0.0),
        tail_wait_quantiles_ms={
            float(key): float(value)
            for key, value in (_as_mapping(mapping.get("tail_wait_quantiles_ms")) or {}).items()
        },
        throughput_work_per_s=(
            None
            if mapping.get("throughput_work_per_s") is None
            else _as_float(mapping.get("throughput_work_per_s"), default=0.0)
        ),
        profile_version=str(mapping.get("profile_version") or "config"),
        metadata=dict(_as_mapping(mapping.get("metadata")) or {}),
    )


def _parse_stage_kind(value: Any) -> DagStageKind:
    if isinstance(value, DagStageKind):
        return value
    normalized = str(value).strip().lower().replace("-", "_")
    try:
        return DagStageKind(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported PARD stage kind: {value!r}") from exc


def _required_mapping(value: Any, label: str) -> dict[str, Any]:
    mapping = _as_mapping(value)
    if mapping is None:
        raise TypeError(f"{label} must be a mapping")
    return mapping


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    items = getattr(value, "items", None)
    if callable(items):
        return dict(items())
    return None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        value_mapping = _as_mapping(value)
        base_mapping = _as_mapping(base_value)
        if value_mapping is not None and base_mapping is not None:
            merged[key] = _deep_merge(base_mapping, value_mapping)
        else:
            merged[key] = value
    return merged


def _required_float(mapping: Mapping[str, Any], key: str) -> float:
    if key not in mapping:
        raise ValueError(f"PARD profile requires {key}")
    return _as_float(mapping[key], default=0.0)


def _as_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _as_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


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
        self.drop_controller = PardDropController(dag_runtime)
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

    async def apply_drop_decision(self, decision: PardBrokerDecision) -> PardDropCleanupResult:
        return await self.drop_controller.apply_drop_decision(decision)

    async def apply_pending_step_boundary(self, request_id: str, stage_id: int) -> PardDropCleanupResult:
        return await self.drop_controller.apply_pending_step_boundary(request_id, stage_id)

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
    "build_pard_runtime_from_stage_pools",
    "planner_mode_for_policy_alias",
]
