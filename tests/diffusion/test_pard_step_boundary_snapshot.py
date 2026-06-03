from types import SimpleNamespace

from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.sched.interface import CachedRequestData
from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput
from vllm_omni.diffusion.worker.utils import RunnerOutput


class _FakeScheduler:

    def __init__(self) -> None:
        self.state = SimpleNamespace(
            req=SimpleNamespace(
                request_id="logical-req",
                request_ids=["external-req"],
            ),
        )

    def get_request_state(self, sched_req_id: str):
        if sched_req_id == "sched-req":
            return self.state
        return None


def test_diffusion_engine_records_step_boundary_snapshot() -> None:
    engine = object.__new__(DiffusionEngine)
    engine.scheduler = _FakeScheduler()

    sched_output = DiffusionSchedulerOutput(
        step_id=9,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(sched_req_ids=["sched-req"]),
        finished_req_ids=set(),
        num_running_reqs=1,
        num_waiting_reqs=0,
    )
    runner_output = BatchRunnerOutput.from_list([
        RunnerOutput(
            req_id="sched-req",
            step_index=3,
            finished=False,
            result=None,
        ),
    ])

    engine._record_step_boundary_snapshot(
        sched_output,
        runner_output,
        {"sched-req"},
    )

    snapshot = engine.get_step_boundary_snapshot()
    assert snapshot["supported"] is True
    assert snapshot["step_id"] == 9
    assert snapshot["scheduled_req_ids"] == ["sched-req"]
    assert snapshot["request_ids"] == ["logical-req", "external-req"]
    assert snapshot["finished_req_ids"] == ["sched-req"]
    assert snapshot["request_outputs"]["sched-req"]["step_index"] == 3
    assert snapshot["request_outputs"]["sched-req"]["request_ids"] == [
        "logical-req",
        "external-req",
    ]
