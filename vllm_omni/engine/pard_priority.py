"""PARD priority queues and workload-intensity switching.

The queue is intentionally independent from the real stage scheduler. Gate C
only defines the PARD decision machinery; Gate D wires it into real execution
after drop/cancel/cleanup is verified.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PardPriorityMode(StrEnum):
    FCFS = "fcfs"
    HBF = "hbf"
    LBF = "lbf"


class PardPriorityPolicy(StrEnum):
    FCFS = "fcfs"
    HBF = "hbf"
    LBF = "lbf"
    ADAPTIVE = "adaptive"
    INSTANT = "instant"


@dataclass(frozen=True, slots=True)
class PardPriorityItem:
    request_id: str
    stage_id: int
    remaining_budget_ms: float | None
    estimated_latency_ms: float
    arrival_seq: int
    enqueued_at_s: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def budget_key(self) -> float:
        if self.remaining_budget_ms is None:
            return float("inf")
        return float(self.remaining_budget_ms)


class PardDoubleEndedPriorityQueue:
    """Double-ended queue ordered by remaining latency budget.

    Low-budget entries are closest to deadline; high-budget entries have the
    most remaining latency budget. FCFS is preserved through arrival_seq.
    """

    def __init__(self) -> None:
        self._items: dict[str, PardPriorityItem] = {}

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._items

    def push(self, item: PardPriorityItem) -> None:
        self._items[item.request_id] = item

    def remove(self, request_id: str) -> PardPriorityItem | None:
        return self._items.pop(request_id, None)

    def pop_low_budget(self) -> PardPriorityItem | None:
        return self._pop_key(lambda item: (item.budget_key, item.arrival_seq, item.request_id))

    def pop_high_budget(self) -> PardPriorityItem | None:
        return self._pop_key(lambda item: (-item.budget_key, item.arrival_seq, item.request_id))

    def pop_fcfs(self) -> PardPriorityItem | None:
        return self._pop_key(lambda item: (item.arrival_seq, item.request_id))

    def peek_low_budget(self) -> PardPriorityItem | None:
        return self._peek_key(lambda item: (item.budget_key, item.arrival_seq, item.request_id))

    def peek_high_budget(self) -> PardPriorityItem | None:
        return self._peek_key(lambda item: (-item.budget_key, item.arrival_seq, item.request_id))

    def pop_for_mode(self, mode: PardPriorityMode) -> PardPriorityItem | None:
        if mode == PardPriorityMode.HBF:
            return self.pop_high_budget()
        if mode == PardPriorityMode.LBF:
            return self.pop_low_budget()
        return self.pop_fcfs()

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "request_id": item.request_id,
                "stage_id": item.stage_id,
                "remaining_budget_ms": item.remaining_budget_ms,
                "estimated_latency_ms": item.estimated_latency_ms,
                "arrival_seq": item.arrival_seq,
            }
            for item in sorted(self._items.values(), key=lambda item: item.arrival_seq)
        ]

    def _peek_key(self, key_fn: Any) -> PardPriorityItem | None:
        if not self._items:
            return None
        return min(self._items.values(), key=key_fn)

    def _pop_key(self, key_fn: Any) -> PardPriorityItem | None:
        item = self._peek_key(key_fn)
        if item is None:
            return None
        return self._items.pop(item.request_id)


@dataclass(frozen=True, slots=True)
class PardWorkloadWindow:
    recent_input_work: float
    smoothed_input_work: float
    stage_service_work: float
    mu: float
    eps: float


@dataclass(frozen=True, slots=True)
class PardPriorityDecision:
    policy: PardPriorityPolicy
    mode: PardPriorityMode
    previous_mode: PardPriorityMode
    window: PardWorkloadWindow
    transition_reason: str


class PardWorkloadIntensityEstimator:
    """PARD workload-intensity switch with delayed transition hysteresis."""

    def __init__(
        self,
        *,
        policy: PardPriorityPolicy = PardPriorityPolicy.ADAPTIVE,
        initial_mode: PardPriorityMode = PardPriorityMode.LBF,
        smoothing_alpha: float = 0.2,
        min_eps: float = 0.05,
        max_eps: float = 0.5,
    ) -> None:
        self.policy = policy
        self.current_mode = initial_mode
        self.smoothing_alpha = min(max(float(smoothing_alpha), 0.0), 1.0)
        self.min_eps = max(float(min_eps), 0.0)
        self.max_eps = max(float(max_eps), self.min_eps)
        self._smoothed_input_work = 0.0

    def decide(
        self,
        *,
        recent_input_work: float,
        stage_service_work: float,
    ) -> PardPriorityDecision:
        recent = max(float(recent_input_work), 0.0)
        service = max(float(stage_service_work), 0.001)
        if self._smoothed_input_work <= 0.0:
            self._smoothed_input_work = recent
        else:
            alpha = self.smoothing_alpha
            self._smoothed_input_work = alpha * recent + (1.0 - alpha) * self._smoothed_input_work
        mu = recent / service
        eps_base = abs(recent - self._smoothed_input_work) / max(self._smoothed_input_work, 1.0)
        eps = min(max(eps_base, self.min_eps), self.max_eps)
        previous = self.current_mode
        mode = previous
        reason = "delayed_transition_hold"
        if self.policy == PardPriorityPolicy.FCFS:
            mode = PardPriorityMode.FCFS
            reason = "fixed_fcfs"
        elif self.policy == PardPriorityPolicy.HBF:
            mode = PardPriorityMode.HBF
            reason = "fixed_hbf"
        elif self.policy == PardPriorityPolicy.LBF:
            mode = PardPriorityMode.LBF
            reason = "fixed_lbf"
        elif self.policy == PardPriorityPolicy.INSTANT:
            if mu > 1.0:
                mode = PardPriorityMode.HBF
                reason = "instant_high_load"
            else:
                mode = PardPriorityMode.LBF
                reason = "instant_normal_load"
        else:
            if mu > 1.0 + eps:
                mode = PardPriorityMode.HBF
                reason = "adaptive_high_load"
            elif mu < 1.0 - eps:
                mode = PardPriorityMode.LBF
                reason = "adaptive_normal_load"
        self.current_mode = mode
        return PardPriorityDecision(
            policy=self.policy,
            mode=mode,
            previous_mode=previous,
            window=PardWorkloadWindow(
                recent_input_work=recent,
                smoothed_input_work=self._smoothed_input_work,
                stage_service_work=service,
                mu=mu,
                eps=eps,
            ),
            transition_reason=reason,
        )


__all__ = [
    "PardDoubleEndedPriorityQueue",
    "PardPriorityDecision",
    "PardPriorityItem",
    "PardPriorityMode",
    "PardPriorityPolicy",
    "PardWorkloadIntensityEstimator",
    "PardWorkloadWindow",
]
