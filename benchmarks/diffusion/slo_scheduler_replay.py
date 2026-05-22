# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import sys
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import SloStepScheduler, StepScheduler
from vllm_omni.diffusion.sched import base_scheduler as base_sched_module
from vllm_omni.diffusion.sched import slo_step_scheduler as slo_sched_module
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def make_synthetic_trace(
    num_requests: int,
    interarrival_s: float,
    slo_scale: float,
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for i in range(num_requests):
        # Alternate compatible and incompatible shapes to expose FIFO head
        # blocking. The larger shape is also more expensive.
        if i % 2 == 0 or i % 6 == 5:
            height = width = 1024
            reference_cost_ms = 900.0
        else:
            height = width = 768
            reference_cost_ms = 520.0
        arrival_s = i * interarrival_s
        slo_ms = reference_cost_ms * slo_scale
        trace.append(
            {
                "request_id": f"req-{i:04d}",
                "height": height,
                "width": width,
                "reference_cost_ms": reference_cost_ms,
                "slo_ms": slo_ms,
                "arrival_s": arrival_s,
                "deadline_s": arrival_s + slo_ms / 1000.0,
            }
        )
    return trace


def build_request(item: dict[str, Any], total_steps: int) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=["prompt-" + str(item["request_id"])],
        sampling_params=OmniDiffusionSamplingParams(
            height=int(item["height"]),
            width=int(item["width"]),
            num_inference_steps=total_steps,
            reference_cost_ms=float(item["reference_cost_ms"]),
            slo_ms=float(item["slo_ms"]),
            arrival_time_s=float(item["arrival_s"]),
            deadline_time_s=float(item["deadline_s"]),
        ),
        request_ids=[str(item["request_id"])],
    )


def scheduled_ids(output: Any) -> list[str]:
    return [req.sched_req_id for req in output.scheduled_new_reqs] + list(
        output.scheduled_cached_reqs.sched_req_ids
    )


def estimate_step_ms(scheduler: Any, req_ids: list[str], batch_growth_alpha: float) -> float:
    single_step_ms = []
    for req_id in req_ids:
        state = scheduler.get_request_state(req_id)
        progress = scheduler._request_progress[req_id]
        single_step_ms.append(float(state.reference_cost_ms) / max(progress.total_steps, 1))
    batch_scale = 1.0 + batch_growth_alpha * max(len(req_ids) - 1, 0)
    return max(single_step_ms) * batch_scale


def run_replay(
    policy_cls: type,
    trace: list[dict[str, Any]],
    *,
    max_num_seqs: int,
    total_steps: int,
    batch_growth_alpha: float,
) -> dict[str, Any]:
    scheduler = policy_cls()
    scheduler.initialize(
        SimpleNamespace(
            max_num_seqs=max_num_seqs,
            additional_config={
                "diffusion_slo_scheduler": {
                    "batch_growth_alpha": batch_growth_alpha,
                }
            },
        )
    )

    next_idx = 0
    now_s = 0.0
    id_to_item: dict[str, dict[str, Any]] = {}
    finish_s: dict[str, float] = {}
    batch_sizes: list[int] = []
    steps_executed = 0

    original_slo_time = slo_sched_module.time.time
    original_base_time = base_sched_module.time.time
    try:
        slo_sched_module.time.time = lambda: now_s
        base_sched_module.time.time = lambda: now_s

        while next_idx < len(trace) or scheduler.has_requests():
            while next_idx < len(trace) and trace[next_idx]["arrival_s"] <= now_s + 1e-12:
                item = trace[next_idx]
                sched_req_id = scheduler.add_request(build_request(item, total_steps))
                id_to_item[sched_req_id] = item
                next_idx += 1

            sched_output = scheduler.schedule()
            req_ids = scheduled_ids(sched_output)
            if not req_ids:
                if next_idx < len(trace):
                    now_s = max(now_s, float(trace[next_idx]["arrival_s"]))
                    continue
                break

            step_ms = estimate_step_ms(scheduler, req_ids, batch_growth_alpha)
            now_s += step_ms / 1000.0
            steps_executed += 1
            batch_sizes.append(len(req_ids))

            runner_outputs = []
            for req_id in req_ids:
                progress = scheduler._request_progress[req_id]
                next_step = progress.current_step + 1
                done = next_step >= progress.total_steps
                runner_outputs.append(
                    RunnerOutput(
                        req_id=req_id,
                        step_index=next_step,
                        finished=done,
                        result=DiffusionOutput(output=None),
                    )
                )
            finished_ids = scheduler.update_from_output(
                sched_output,
                BatchRunnerOutput.from_list(runner_outputs),
            )
            for req_id in finished_ids:
                finish_s[req_id] = now_s
    finally:
        slo_sched_module.time.time = original_slo_time
        base_sched_module.time.time = original_base_time

    per_request = []
    for req_id, item in id_to_item.items():
        done_s = finish_s.get(req_id)
        if done_s is None:
            continue
        e2e_s = done_s - float(item["arrival_s"])
        per_request.append(
            {
                "request_id": req_id,
                "shape": "{}x{}".format(item["height"], item["width"]),
                "arrival_s": item["arrival_s"],
                "deadline_s": item["deadline_s"],
                "finish_s": done_s,
                "e2e_s": e2e_s,
                "slo_met": done_s <= float(item["deadline_s"]),
            }
        )

    e2e_values = [row["e2e_s"] for row in per_request]
    met = sum(1 for row in per_request if row["slo_met"])
    return {
        "policy": policy_cls.__name__,
        "requests": len(per_request),
        "slo_attainment": met / len(per_request) if per_request else 0.0,
        "slo_misses": len(per_request) - met,
        "makespan_s": now_s,
        "mean_e2e_s": statistics.mean(e2e_values) if e2e_values else 0.0,
        "p95_e2e_s": statistics.quantiles(e2e_values, n=100)[94] if len(e2e_values) >= 2 else 0.0,
        "mean_batch_size": statistics.mean(batch_sizes) if batch_sizes else 0.0,
        "p95_batch_size": statistics.quantiles(batch_sizes, n=100)[94] if len(batch_sizes) >= 2 else 0.0,
        "steps_executed": steps_executed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a synthetic DiT SLO workload against diffusion schedulers.")
    parser.add_argument("--num-requests", type=int, default=120)
    parser.add_argument("--interarrival-s", type=float, default=0.08)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--slo-scale", type=float, default=3.0)
    parser.add_argument("--batch-growth-alpha", type=float, default=0.60)
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()

    trace = make_synthetic_trace(args.num_requests, args.interarrival_s, args.slo_scale)
    results = [
        run_replay(
            StepScheduler,
            trace,
            max_num_seqs=args.max_num_seqs,
            total_steps=args.steps,
            batch_growth_alpha=args.batch_growth_alpha,
        ),
        run_replay(
            SloStepScheduler,
            trace,
            max_num_seqs=args.max_num_seqs,
            total_steps=args.steps,
            batch_growth_alpha=args.batch_growth_alpha,
        ),
    ]
    payload = {
        "config": vars(args),
        "results": results,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
