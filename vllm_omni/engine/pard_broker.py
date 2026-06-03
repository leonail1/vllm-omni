"""PARD Request Broker.

Gate C keeps the broker as a control-plane decision module. It can mark a DAG
request as dropped and maintain the DEPQ, but it is not wired into real
StagePool cancellation until Gate D.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from vllm_omni.engine.dag_types import DagRequestContext, DagTraceEvent
from vllm_omni.engine.pard_planner import PardLatencyPlan, PardStatePlanner
from vllm_omni.engine.pard_priority import (
    PardDoubleEndedPriorityQueue,
    PardPriorityDecision,
    PardPriorityItem,
    PardPriorityMode,
    PardPriorityPolicy,
    PardWorkloadIntensityEstimator,
)


class PardBrokerDecisionKind(StrEnum):
    ACCEPT = "accept"
    PROACTIVE_DROP = "proactive_drop"
    REACTIVE_DROP = "reactive_drop"


@dataclass(frozen=True, slots=True)
class PardBrokerPolicy:
    policy_name: str = "dag_full_pard"
    paper_name: str = "full PARD"
    ablation_semantic: str = "full_pard"
    proactive_drop: bool = True
    reactive_drop: bool = True
    priority_policy: PardPriorityPolicy = PardPriorityPolicy.ADAPTIVE
    stage_budget_mode: str = "e2e"
    overload_control: bool = False
    overload_queue_delay_threshold_ms: float = 100.0
    overload_admit_fraction: float = 0.5


@dataclass(frozen=True, slots=True)
class PardBrokerDecision:
    kind: PardBrokerDecisionKind
    request_id: str
    stage_id: int
    latency_plan: PardLatencyPlan
    priority_decision: PardPriorityDecision
    reason: str
    invalid_compute_ms: float = 0.0
    queue_size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.kind == PardBrokerDecisionKind.ACCEPT

    @property
    def dropped(self) -> bool:
        return self.kind in {
            PardBrokerDecisionKind.PROACTIVE_DROP,
            PardBrokerDecisionKind.REACTIVE_DROP,
        }


class PardRequestBroker:
    """Stage-entry PARD broker with DEPQ-backed priority admission."""

    def __init__(
        self,
        *,
        stage_id: int,
        planner: PardStatePlanner,
        policy: PardBrokerPolicy,
        service_work_per_window: float,
        initial_priority_mode: PardPriorityMode = PardPriorityMode.LBF,
    ) -> None:
        self.stage_id = stage_id
        self.planner = planner
        self.policy = policy
        self.queue = PardDoubleEndedPriorityQueue()
        self.priority = PardWorkloadIntensityEstimator(
            policy=policy.priority_policy,
            initial_mode=initial_priority_mode,
        )
        self.service_work_per_window = max(float(service_work_per_window), 0.001)
        self._arrival_seq = 0
        self._attempt_seq = 0
        self.decisions: list[PardBrokerDecision] = []

    def stage_entry(
        self,
        ctx: DagRequestContext,
        *,
        now_s: float | None = None,
        recent_input_work: float = 0.0,
        stage_snapshots: Mapping[int, Mapping[str, Any]] | None = None,
    ) -> PardBrokerDecision:
        now = time.time() if now_s is None else now_s
        self._attempt_seq += 1
        plan = self.planner.plan_stage_entry(
            ctx,
            self.stage_id,
            now_s=now,
            stage_snapshots=stage_snapshots,
        )
        priority_decision = self.priority.decide(
            recent_input_work=recent_input_work,
            stage_service_work=self.service_work_per_window,
        )
        if ctx.dropped:
            decision = self._already_dropped_decision(
                ctx,
                plan=plan,
                priority_decision=priority_decision,
                now_s=now,
            )
            self.decisions.append(decision)
            return decision
        if self.policy.reactive_drop and _deadline_already_missed(ctx, now):
            decision = self._drop_decision(
                ctx,
                plan=plan,
                priority_decision=priority_decision,
                kind=PardBrokerDecisionKind.REACTIVE_DROP,
                reason="deadline_already_missed",
                now_s=now,
            )
            self.decisions.append(decision)
            return decision
        if self.policy.overload_control and self._should_overload_drop(plan):
            decision = self._drop_decision(
                ctx,
                plan=plan,
                priority_decision=priority_decision,
                kind=PardBrokerDecisionKind.PROACTIVE_DROP,
                reason="overload_control_queue_delay",
                now_s=now,
            )
            self.decisions.append(decision)
            return decision
        budget_reason = self._stage_budget_drop_reason(plan)
        if budget_reason is not None:
            decision = self._drop_decision(
                ctx,
                plan=plan,
                priority_decision=priority_decision,
                kind=PardBrokerDecisionKind.PROACTIVE_DROP,
                reason=budget_reason,
                now_s=now,
            )
            self.decisions.append(decision)
            return decision
        if self.policy.proactive_drop and plan.would_miss_deadline:
            decision = self._drop_decision(
                ctx,
                plan=plan,
                priority_decision=priority_decision,
                kind=PardBrokerDecisionKind.PROACTIVE_DROP,
                reason="predicted_e2e_slo_miss",
                now_s=now,
            )
            self.decisions.append(decision)
            return decision
        self._arrival_seq += 1
        self.queue.push(
            PardPriorityItem(
                request_id=ctx.request_id,
                stage_id=self.stage_id,
                remaining_budget_ms=plan.remaining_budget_ms,
                estimated_latency_ms=plan.l_cur_ms + plan.l_sub_ms,
                arrival_seq=self._arrival_seq,
                enqueued_at_s=now,
                metadata={
                    "broker_policy_name": self.policy.policy_name,
                    "paper_name": self.policy.paper_name,
                    "ablation_semantic": self.policy.ablation_semantic,
                    "priority_mode": priority_decision.mode.value,
                    "planner_mode": plan.planner_mode.value,
                },
            )
        )
        decision = PardBrokerDecision(
            kind=PardBrokerDecisionKind.ACCEPT,
            request_id=ctx.request_id,
            stage_id=self.stage_id,
            latency_plan=plan,
            priority_decision=priority_decision,
            reason="accepted",
            queue_size=len(self.queue),
            metadata={
                "broker_policy_name": self.policy.policy_name,
                "paper_name": self.policy.paper_name,
                "ablation_semantic": self.policy.ablation_semantic,
                "stage_budget_mode": self.policy.stage_budget_mode,
            },
        )
        _append_trace(ctx, decision, now_s=now)
        self.decisions.append(decision)
        return decision

    def pop_next(self, *, recent_input_work: float = 0.0) -> PardPriorityItem | None:
        priority_decision = self.priority.decide(
            recent_input_work=recent_input_work,
            stage_service_work=self.service_work_per_window,
        )
        return self.queue.pop_for_mode(priority_decision.mode)

    def remove(self, request_id: str) -> PardPriorityItem | None:
        return self.queue.remove(request_id)

    def snapshot(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "policy_name": self.policy.policy_name,
            "queue_size": len(self.queue),
            "queue": self.queue.snapshot(),
            "decisions": [
                {
                    "kind": decision.kind.value,
                    "request_id": decision.request_id,
                    "reason": decision.reason,
                    "priority_mode": decision.priority_decision.mode.value,
                    "mu": decision.priority_decision.window.mu,
                    "eps": decision.priority_decision.window.eps,
                    "estimated_e2e_latency_ms": decision.latency_plan.estimated_e2e_latency_ms,
                    "remaining_budget_ms": decision.latency_plan.remaining_budget_ms,
                    "invalid_compute_ms": decision.invalid_compute_ms,
                }
                for decision in self.decisions
            ],
        }

    def _drop_decision(
        self,
        ctx: DagRequestContext,
        *,
        plan: PardLatencyPlan,
        priority_decision: PardPriorityDecision,
        kind: PardBrokerDecisionKind,
        reason: str,
        now_s: float,
    ) -> PardBrokerDecision:
        invalid_compute_ms = _invalid_compute_ms(ctx)
        ctx.mark_dropped(self.stage_id, reason, now_s=now_s)
        self.queue.remove(ctx.request_id)
        decision = PardBrokerDecision(
            kind=kind,
            request_id=ctx.request_id,
            stage_id=self.stage_id,
            latency_plan=plan,
            priority_decision=priority_decision,
            reason=reason,
            invalid_compute_ms=invalid_compute_ms,
            queue_size=len(self.queue),
            metadata={
                "broker_policy_name": self.policy.policy_name,
                "paper_name": self.policy.paper_name,
                "ablation_semantic": self.policy.ablation_semantic,
                "stage_budget_mode": self.policy.stage_budget_mode,
            },
        )
        _append_trace(ctx, decision, now_s=now_s)
        return decision

    def _already_dropped_decision(
        self,
        ctx: DagRequestContext,
        *,
        plan: PardLatencyPlan,
        priority_decision: PardPriorityDecision,
        now_s: float,
    ) -> PardBrokerDecision:
        invalid_compute_ms = _invalid_compute_ms(ctx)
        self.queue.remove(ctx.request_id)
        original_drop_stage_id = ctx.drop_stage_id
        original_drop_reason = ctx.drop_reason
        decision = PardBrokerDecision(
            kind=PardBrokerDecisionKind.REACTIVE_DROP,
            request_id=ctx.request_id,
            stage_id=self.stage_id,
            latency_plan=plan,
            priority_decision=priority_decision,
            reason="request_already_dropped",
            invalid_compute_ms=invalid_compute_ms,
            queue_size=len(self.queue),
            metadata={
                "broker_policy_name": self.policy.policy_name,
                "paper_name": self.policy.paper_name,
                "ablation_semantic": self.policy.ablation_semantic,
                "stage_budget_mode": self.policy.stage_budget_mode,
                "original_drop_stage_id": original_drop_stage_id,
                "original_drop_reason": original_drop_reason,
            },
        )
        _append_trace(ctx, decision, now_s=now_s)
        return decision

    def _should_overload_drop(self, plan: PardLatencyPlan) -> bool:
        if plan.current_stage.queue_wait_ms < self.policy.overload_queue_delay_threshold_ms:
            return False
        admit_fraction = min(max(float(self.policy.overload_admit_fraction), 0.0), 1.0)
        if admit_fraction <= 0.0:
            return True
        if admit_fraction >= 1.0:
            return False
        admit_every = max(int(round(1.0 / admit_fraction)), 1)
        return self._attempt_seq % admit_every != 0

    def _stage_budget_drop_reason(self, plan: PardLatencyPlan) -> str | None:
        if not self.policy.proactive_drop or plan.slo_budget_ms is None:
            return None
        if self.policy.stage_budget_mode == "fixed_split":
            stage_count = max(len(self.planner.config.stage_specs), 1)
            stage_budget_ms = plan.slo_budget_ms / stage_count
            if plan.l_pre_ms + plan.l_cur_ms > stage_budget_ms:
                return "fixed_split_stage_budget_exceeded"
        elif self.policy.stage_budget_mode == "wcl":
            wcl_budget_ms = _wcl_stage_budget_ms(plan, self.planner.config.stage_specs)
            if wcl_budget_ms is not None and plan.l_pre_ms + plan.l_cur_ms > wcl_budget_ms:
                return "wcl_stage_budget_exceeded"
        return None


def policy_for_alias(alias: str) -> PardBrokerPolicy:
    normalized = alias.strip().lower()
    mapping: dict[str, PardBrokerPolicy] = {
        "dag_no_drop": PardBrokerPolicy(
            policy_name="dag_no_drop",
            paper_name="no-drop",
            ablation_semantic="disable_pard_drop",
            proactive_drop=False,
            reactive_drop=False,
            priority_policy=PardPriorityPolicy.FCFS,
        ),
        "dag_reactive_drop": PardBrokerPolicy(
            policy_name="dag_reactive_drop",
            paper_name="reactive",
            ablation_semantic="reactive_drop_only",
            proactive_drop=False,
            reactive_drop=True,
            priority_policy=PardPriorityPolicy.FCFS,
        ),
        "dag_pard_back": PardBrokerPolicy(
            policy_name="dag_pard_back",
            paper_name="PARD-back",
            ablation_semantic="ignore_subsequent_modules_lsub_zero",
        ),
        "dag_pard_sf": PardBrokerPolicy(
            policy_name="dag_pard_sf",
            paper_name="PARD-sf",
            ablation_semantic="subsequent_execution_only",
        ),
        "dag_pard_oc": PardBrokerPolicy(
            policy_name="dag_pard_oc",
            paper_name="PARD-oc",
            ablation_semantic="overload_control_queue_delay_threshold",
            overload_control=True,
        ),
        "dag_pard_split": PardBrokerPolicy(
            policy_name="dag_pard_split",
            paper_name="PARD-split",
            ablation_semantic="fixed_per_module_slo_split",
            stage_budget_mode="fixed_split",
        ),
        "dag_pard_wcl": PardBrokerPolicy(
            policy_name="dag_pard_wcl",
            paper_name="PARD-WCL",
            ablation_semantic="worst_case_latency_dynamic_budget",
            stage_budget_mode="wcl",
        ),
        "dag_pard_lower": PardBrokerPolicy(
            policy_name="dag_pard_lower",
            paper_name="PARD-lower",
            ablation_semantic="lower_bound_subsequent_batch_wait",
        ),
        "dag_pard_upper": PardBrokerPolicy(
            policy_name="dag_pard_upper",
            paper_name="PARD-upper",
            ablation_semantic="upper_bound_subsequent_batch_wait",
        ),
        "dag_pard_fcfs": PardBrokerPolicy(
            policy_name="dag_pard_fcfs",
            paper_name="PARD-FCFS",
            ablation_semantic="fcfs_priority",
            priority_policy=PardPriorityPolicy.FCFS,
        ),
        "dag_pard_hbf": PardBrokerPolicy(
            policy_name="dag_pard_hbf",
            paper_name="PARD-HBF",
            ablation_semantic="fixed_high_budget_first",
            priority_policy=PardPriorityPolicy.HBF,
        ),
        "dag_pard_lbf": PardBrokerPolicy(
            policy_name="dag_pard_lbf",
            paper_name="PARD-LBF",
            ablation_semantic="fixed_low_budget_first",
            priority_policy=PardPriorityPolicy.LBF,
        ),
        "dag_pard_instant": PardBrokerPolicy(
            policy_name="dag_pard_instant",
            paper_name="PARD-instant",
            ablation_semantic="instant_hbf_lbf_transition",
            priority_policy=PardPriorityPolicy.INSTANT,
        ),
        "dag_full_pard": PardBrokerPolicy(
            policy_name="dag_full_pard",
            paper_name="full PARD",
            ablation_semantic="full_pard",
            priority_policy=PardPriorityPolicy.ADAPTIVE,
        ),
        "dag_full_pard_step_preemptive": PardBrokerPolicy(
            policy_name="dag_full_pard_step_preemptive",
            paper_name="full PARD + step preemption",
            ablation_semantic="full_pard_with_dit_step_boundary_preemption",
            priority_policy=PardPriorityPolicy.ADAPTIVE,
        ),
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported PARD policy alias: {alias}")
    return mapping[normalized]


def _deadline_already_missed(ctx: DagRequestContext, now_s: float) -> bool:
    return ctx.deadline_time_s is not None and now_s >= ctx.deadline_time_s


def _invalid_compute_ms(ctx: DagRequestContext) -> float:
    total = 0.0
    for lifecycle in ctx.stage_lifecycle.values():
        if lifecycle.execution_ms is not None:
            total += max(float(lifecycle.execution_ms), 0.0)
    return total


def _append_trace(ctx: DagRequestContext, decision: PardBrokerDecision, *, now_s: float) -> None:
    window = decision.priority_decision.window
    ctx.trace.append(
        DagTraceEvent(
            request_id=ctx.request_id,
            event="pard_broker_decision",
            stage_id=decision.stage_id,
            timestamp_s=now_s,
            details={
                "kind": decision.kind.value,
                "reason": decision.reason,
                "policy_name": decision.metadata.get("broker_policy_name"),
                "paper_name": decision.metadata.get("paper_name"),
                "ablation_semantic": decision.metadata.get("ablation_semantic"),
                "priority_policy": decision.priority_decision.policy.value,
                "priority_mode": decision.priority_decision.mode.value,
                "transition_reason": decision.priority_decision.transition_reason,
                "recent_input_work": window.recent_input_work,
                "smoothed_input_work": window.smoothed_input_work,
                "stage_service_work": window.stage_service_work,
                "mu": window.mu,
                "eps": window.eps,
                "l_pre_ms": decision.latency_plan.l_pre_ms,
                "l_cur_ms": decision.latency_plan.l_cur_ms,
                "l_sub_ms": decision.latency_plan.l_sub_ms,
                "estimated_e2e_latency_ms": decision.latency_plan.estimated_e2e_latency_ms,
                "remaining_budget_ms": decision.latency_plan.remaining_budget_ms,
                "current_stage_profile_version": decision.latency_plan.current_stage.profile_version,
                "current_stage_estimate_source": decision.latency_plan.current_stage.source,
                "original_drop_stage_id": decision.metadata.get("original_drop_stage_id"),
                "original_drop_reason": decision.metadata.get("original_drop_reason"),
                "invalid_compute_ms": decision.invalid_compute_ms,
                "queue_size": decision.queue_size,
            },
        )
    )


def _wcl_stage_budget_ms(plan: PardLatencyPlan, stage_specs: tuple[Any, ...]) -> float | None:
    if plan.slo_budget_ms is None:
        return None
    current_wcl = plan.current_stage.current_total_ms + plan.current_stage.tail_quantile_ms
    downstream_wcl = sum(max(value, 0.0) for value in plan.downstream_path_ms_by_stage.values())
    total_wcl = max(current_wcl + downstream_wcl, 0.001)
    stage_count = max(len(stage_specs), 1)
    # WCL budget allocation uses the stage's worst-case share, falling back to
    # a fixed split only when every profile reports zero cost.
    if total_wcl <= 0.001:
        return plan.slo_budget_ms / stage_count
    return plan.slo_budget_ms * (current_wcl / total_wcl)


__all__ = [
    "PardBrokerDecision",
    "PardBrokerDecisionKind",
    "PardBrokerPolicy",
    "PardRequestBroker",
    "policy_for_alias",
]
