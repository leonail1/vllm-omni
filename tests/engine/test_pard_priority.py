from __future__ import annotations

import pytest

from vllm_omni.engine.pard_priority import (
    PardDoubleEndedPriorityQueue,
    PardPriorityItem,
    PardPriorityMode,
    PardPriorityPolicy,
    PardWorkloadIntensityEstimator,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_depq_pops_low_budget_high_budget_and_fcfs() -> None:
    queue = PardDoubleEndedPriorityQueue()
    queue.push(PardPriorityItem("req-mid", 0, 50.0, 20.0, 2))
    queue.push(PardPriorityItem("req-low", 0, 10.0, 20.0, 3))
    queue.push(PardPriorityItem("req-high", 0, 100.0, 20.0, 1))

    assert queue.peek_low_budget().request_id == "req-low"
    assert queue.peek_high_budget().request_id == "req-high"
    assert queue.pop_low_budget().request_id == "req-low"
    assert queue.pop_high_budget().request_id == "req-high"
    assert queue.pop_fcfs().request_id == "req-mid"
    assert queue.pop_fcfs() is None


def test_depq_mode_pop_uses_pard_priority_mode() -> None:
    queue = PardDoubleEndedPriorityQueue()
    queue.push(PardPriorityItem("req-a", 0, 5.0, 10.0, 1))
    queue.push(PardPriorityItem("req-b", 0, 50.0, 10.0, 2))

    assert queue.pop_for_mode(PardPriorityMode.HBF).request_id == "req-b"
    assert queue.pop_for_mode(PardPriorityMode.LBF).request_id == "req-a"


def test_adaptive_priority_uses_hbf_lbf_and_delayed_transition() -> None:
    estimator = PardWorkloadIntensityEstimator(policy=PardPriorityPolicy.ADAPTIVE)

    high = estimator.decide(recent_input_work=200.0, stage_service_work=100.0)
    hold = estimator.decide(recent_input_work=103.0, stage_service_work=100.0)
    low = estimator.decide(recent_input_work=20.0, stage_service_work=100.0)

    assert high.mode == PardPriorityMode.HBF
    assert high.transition_reason == "adaptive_high_load"
    assert hold.mode == PardPriorityMode.HBF
    assert hold.transition_reason == "delayed_transition_hold"
    assert low.mode == PardPriorityMode.LBF
    assert low.transition_reason == "adaptive_normal_load"


def test_instant_priority_skips_delayed_transition() -> None:
    estimator = PardWorkloadIntensityEstimator(policy=PardPriorityPolicy.INSTANT)

    high = estimator.decide(recent_input_work=101.0, stage_service_work=100.0)
    low = estimator.decide(recent_input_work=99.0, stage_service_work=100.0)

    assert high.mode == PardPriorityMode.HBF
    assert high.transition_reason == "instant_high_load"
    assert low.mode == PardPriorityMode.LBF
    assert low.transition_reason == "instant_normal_load"


@pytest.mark.parametrize(
    ("policy", "expected_mode", "expected_reason"),
    [
        (PardPriorityPolicy.FCFS, PardPriorityMode.FCFS, "fixed_fcfs"),
        (PardPriorityPolicy.HBF, PardPriorityMode.HBF, "fixed_hbf"),
        (PardPriorityPolicy.LBF, PardPriorityMode.LBF, "fixed_lbf"),
    ],
)
def test_fixed_priority_policies(policy, expected_mode, expected_reason) -> None:
    estimator = PardWorkloadIntensityEstimator(policy=policy)

    decision = estimator.decide(recent_input_work=1000.0, stage_service_work=1.0)

    assert decision.mode == expected_mode
    assert decision.transition_reason == expected_reason
