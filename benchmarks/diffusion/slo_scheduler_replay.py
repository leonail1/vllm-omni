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
from vllm_omni.diffusion.sched.step_cost_model import (
    DiffusionStepCostModel,
    estimate_request_effective_size,
)
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def make_synthetic_trace(
    num_requests: int,
    interarrival_s: float,
    slo_scale: float,
    *,
    workload: str = "interleaved",
    shape_order: str = "interleaved",
    batch_hint: int = 4,
    large_period: int | None = None,
    tight_slo_scale: float = 2.2,
    loose_slo_scale: float = 5.0,
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for i in range(num_requests):
        per_request_slo_scale = slo_scale
        if workload == "mixed_slo":
            # One loose large request followed by several tighter small
            # requests. FIFO tends to keep serving the large head request,
            # while SLO-aware scheduling should rescue the tight bucket.
            use_large = i % max(large_period or batch_hint, 1) == 0
            per_request_slo_scale = loose_slo_scale if use_large else tight_slo_scale
        elif shape_order == "batch_friendly":
            use_large = (i // max(batch_hint, 1)) % 2 == 0
        else:
            use_large = i % 2 == 0 or i % 6 == 5

        # Interleaving compatible and incompatible shapes exposes FIFO head
        # blocking. Batch-friendly order is useful as a sanity baseline.
        if use_large:
            height = width = 1024
            reference_cost_ms = 900.0
        else:
            height = width = 768
            reference_cost_ms = 520.0
        arrival_s = i * interarrival_s
        slo_ms = reference_cost_ms * per_request_slo_scale
        trace.append(
            {
                "request_id": f"req-{i:04d}",
                "height": height,
                "width": width,
                "reference_cost_ms": reference_cost_ms,
                "slo_ms": slo_ms,
                "slo_multiplier": per_request_slo_scale,
                "arrival_s": arrival_s,
                "deadline_s": arrival_s + slo_ms / 1000.0,
            }
        )
    return trace


def populate_profiled_costs(
    trace: list[dict[str, Any]],
    *,
    cost_model: DiffusionStepCostModel | None,
    model: str,
    total_steps: int,
    slo_scale: float,
) -> list[dict[str, Any]]:
    if cost_model is None:
        return trace
    out: list[dict[str, Any]] = []
    for item in trace:
        estimate = cost_model.estimate(
            model=model,
            width=item["width"],
            height=item["height"],
            num_frames=1,
            batch_size=1,
            effective_batch_size=1.0,
        )
        if estimate.source == "default":
            out.append(dict(item))
            continue
        reference_cost_ms = estimate.step_ms * total_steps
        request_slo_scale = float(item.get("slo_multiplier", slo_scale))
        cloned = dict(item)
        cloned["reference_cost_ms"] = reference_cost_ms
        cloned["slo_ms"] = reference_cost_ms * request_slo_scale
        cloned["deadline_s"] = float(item["arrival_s"]) + cloned["slo_ms"] / 1000.0
        out.append(cloned)
    return out


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
    return estimate_step_ms_with_model(scheduler, req_ids, batch_growth_alpha, cost_model=None, model="")


def estimate_step_ms_with_model(
    scheduler: Any,
    req_ids: list[str],
    batch_growth_alpha: float,
    *,
    cost_model: DiffusionStepCostModel | None,
    model: str,
) -> float:
    states = [scheduler.get_request_state(req_id) for req_id in req_ids]
    states = [state for state in states if state is not None]
    if states and cost_model is not None:
        sampling = states[0].req.sampling_params
        height = getattr(sampling, "height", None)
        width = getattr(sampling, "width", None)
        estimate = cost_model.estimate(
            model=model,
            width=width,
            height=height,
            num_frames=getattr(sampling, "num_frames", 1),
            batch_size=len(states),
            effective_batch_size=sum(_request_effective_size(state) for state in states),
        )
        if estimate.source != "default":
            return estimate.step_ms

    single_step_ms = []
    for req_id in req_ids:
        state = scheduler.get_request_state(req_id)
        progress = scheduler._request_progress[req_id]
        single_step_ms.append(float(state.reference_cost_ms) / max(progress.total_steps, 1))
    batch_scale = 1.0 + batch_growth_alpha * max(len(req_ids) - 1, 0)
    return max(single_step_ms) * batch_scale


def _request_effective_size(state: Any) -> float:
    prompts = getattr(state.req, "prompts", None)
    prompt_count = float(max(len(prompts), 1)) if isinstance(prompts, (list, tuple)) else 1.0
    return prompt_count * estimate_request_effective_size(state.req.sampling_params, prompts=prompts)


def run_replay(
    policy_cls: type,
    trace: list[dict[str, Any]],
    *,
    max_num_seqs: int,
    total_steps: int,
    batch_growth_alpha: float,
    cost_model: DiffusionStepCostModel | None,
    model: str,
    slo_config: dict[str, Any],
) -> dict[str, Any]:
    scheduler = policy_cls()
    scheduler.initialize(
        SimpleNamespace(
            model=model,
            max_num_seqs=max_num_seqs,
            additional_config={
                "diffusion_slo_scheduler": slo_config,
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

            step_ms = estimate_step_ms_with_model(
                scheduler,
                req_ids,
                batch_growth_alpha,
                cost_model=cost_model,
                model=model,
            )
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
    by_shape: dict[str, dict[str, int]] = {}
    for row in per_request:
        shape = str(row["shape"])
        stats = by_shape.setdefault(shape, {"requests": 0, "slo_misses": 0})
        stats["requests"] += 1
        stats["slo_misses"] += 0 if row["slo_met"] else 1
    return {
        "policy": policy_cls.__name__,
        "requests": len(per_request),
        "slo_attainment": met / len(per_request) if per_request else 0.0,
        "slo_misses": len(per_request) - met,
        "slo_misses_by_shape": by_shape,
        "makespan_s": now_s,
        "throughput_rps": (len(per_request) / now_s) if now_s > 0 else 0.0,
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
    parser.add_argument("--workload", choices=["interleaved", "mixed_slo"], default="interleaved")
    parser.add_argument("--shape-order", choices=["interleaved", "batch_friendly"], default="interleaved")
    parser.add_argument("--large-period", type=int, default=None)
    parser.add_argument("--tight-slo-scale", type=float, default=2.2)
    parser.add_argument("--loose-slo-scale", type=float, default=5.0)
    parser.add_argument("--model", type=str, default="Qwen/Qwen-Image")
    parser.add_argument("--step-cost-model-path", type=str, default=None)
    parser.add_argument("--step-cost-metric", type=str, default="p90_ms")
    parser.add_argument("--step-cost-formula", type=str, default=None)
    parser.add_argument("--step-cost-safety-factor", type=float, default=1.0)
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()

    slo_config: dict[str, Any] = {
        "batch_growth_alpha": args.batch_growth_alpha,
    }
    if args.step_cost_model_path:
        slo_config["step_cost_model_path"] = args.step_cost_model_path
        slo_config["step_cost_metric"] = args.step_cost_metric
    if args.step_cost_formula:
        slo_config["step_cost_formula"] = args.step_cost_formula
    if args.step_cost_safety_factor:
        slo_config["step_cost_safety_factor"] = args.step_cost_safety_factor

    cost_model = DiffusionStepCostModel.from_config(slo_config, default_step_ms=1.0)
    trace = make_synthetic_trace(
        args.num_requests,
        args.interarrival_s,
        args.slo_scale,
        workload=args.workload,
        shape_order=args.shape_order,
        batch_hint=args.max_num_seqs,
        large_period=args.large_period,
        tight_slo_scale=args.tight_slo_scale,
        loose_slo_scale=args.loose_slo_scale,
    )
    trace = populate_profiled_costs(
        trace,
        cost_model=cost_model,
        model=args.model,
        total_steps=args.steps,
        slo_scale=args.slo_scale,
    )
    results = [
        run_replay(
            StepScheduler,
            trace,
            max_num_seqs=args.max_num_seqs,
            total_steps=args.steps,
            batch_growth_alpha=args.batch_growth_alpha,
            cost_model=cost_model,
            model=args.model,
            slo_config=slo_config,
        ),
        run_replay(
            SloStepScheduler,
            trace,
            max_num_seqs=args.max_num_seqs,
            total_steps=args.steps,
            batch_growth_alpha=args.batch_growth_alpha,
            cost_model=cost_model,
            model=args.model,
            slo_config=slo_config,
        ),
    ]
    payload = {
        "config": vars(args),
        "cost_model_enabled": cost_model is not None,
        "results": results,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
