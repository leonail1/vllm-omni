# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL = "Qwen/Qwen-Image"
DEFAULT_COST_MODEL = "benchmarks/diffusion/profile_results/latent_token_aspect/step_cost_model.json"
WORKLOAD_SHAPES = {
    "current-mix": [(512, 512), (768, 768), (512, 512), (768, 768), (1024, 1024)],
    "large-heavy": [(1024, 1024), (1024, 1024), (768, 768), (1024, 1024), (512, 512)],
    "rectangular-mix": [
        (512, 768),
        (768, 512),
        (576, 1024),
        (1024, 576),
        (512, 512),
        (768, 768),
        (1024, 1024),
    ],
    "bursty": [(512, 512), (768, 768), (512, 512), (768, 768), (1024, 1024)],
}


def _imagenet_labels(dataset_readme: str | Path) -> list[str]:
    text = Path(dataset_readme).read_text(encoding="utf-8")
    labels: list[str] = []
    for line in text.splitlines():
        match = re.match(r"\s*'\d+':\s*(.+)$", line)
        if match:
            label = match.group(1).strip().split(",")[0].strip()
            if label:
                labels.append(label)
    return labels or ["object"]


def _prompt(labels: list[str], index: int) -> str:
    label = labels[index % len(labels)]
    templates = [
        "a high quality photo of a {label}, detailed, natural lighting",
        "a realistic documentary photograph showing a {label} in its natural environment",
        "a sharp studio image of a {label}, clear composition and rich detail",
        "a photo-realistic scene centered on a {label}, balanced lighting and texture",
    ]
    return templates[index % len(templates)].format(label=label)


def _request_repr(row: dict[str, Any]) -> str:
    fields = ", ".join(f"{key}={value!r}" for key, value in row.items())
    return f"Request({fields})"


def _shape_for_workload(workload: str, index: int) -> tuple[int, int]:
    try:
        shapes = WORKLOAD_SHAPES[workload]
    except KeyError as exc:
        supported = ", ".join(sorted(WORKLOAD_SHAPES))
        raise ValueError(f"unsupported workload {workload!r}; choose one of: {supported}") from exc
    return shapes[index % len(shapes)]


def _arrival_time_s(args: argparse.Namespace, index: int) -> float:
    if args.workload != "bursty":
        return index * args.global_interarrival_s
    burst_size = max(args.burst_size, 1)
    burst_index = index // burst_size
    in_burst_index = index % burst_size
    burst_span_s = args.burst_interarrival_s * max(burst_size - 1, 0)
    burst_period_s = max(args.global_interarrival_s * burst_size, burst_span_s)
    return burst_index * burst_period_s + in_burst_index * args.burst_interarrival_s


def make_trace(args: argparse.Namespace) -> None:
    from vllm_omni.diffusion.sched.step_cost_model import DiffusionStepCostModel

    if args.workload not in WORKLOAD_SHAPES:
        supported = ", ".join(sorted(WORKLOAD_SHAPES))
        raise ValueError(f"unsupported workload {args.workload!r}; choose one of: {supported}")
    labels = _imagenet_labels(args.imagenet_readme)
    cost_model = DiffusionStepCostModel.from_config(
        {
            "step_cost_model_path": args.step_cost_model_path,
            "step_cost_metric": args.step_cost_metric,
            "step_cost_formula": args.step_cost_formula,
        },
        default_step_ms=1.0,
    )
    if cost_model is None:
        raise RuntimeError("step cost model did not load")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_rows: list[dict[str, Any]] = []
    with output.open("w", encoding="utf-8") as f:
        for i in range(args.num_requests):
            width, height = _shape_for_workload(args.workload, i)
            estimate = cost_model.estimate(
                model=args.model,
                width=width,
                height=height,
                num_frames=1,
                batch_size=1,
                effective_batch_size=1.0,
            )
            reference_cost_ms = estimate.step_ms * args.num_inference_steps
            slo_ms = reference_cost_ms * args.slo_scale
            arrival_s = _arrival_time_s(args, i)
            row = {
                "request_id": f"{args.trace_id}-{i:04d}",
                "prompt": _prompt(labels, i),
                "width": width,
                "height": height,
                "num_inference_steps": args.num_inference_steps,
                "seed": args.seed + i,
                "timestamp": arrival_s,
                "arrival_time_s": arrival_s,
                "deadline_time_s": arrival_s + slo_ms / 1000.0,
                "reference_cost_ms": reference_cost_ms,
                "slo_ms": slo_ms,
                "extra_body": {
                    "extra_args": {
                        "profile_mode": args.profile_mode,
                        "profile_policy": args.profile_policy,
                        "profile_scale": args.slo_scale,
                        "profile_repeat": args.profile_repeat,
                        "profile_trace_id": args.trace_id,
                        "profile_workload": args.workload,
                        "profile_shape": f"{width}x{height}",
                    }
                },
            }
            f.write(_request_repr(row) + "\n")
            metadata_rows.append(
                {
                    "request_id": row["request_id"],
                    "workload": args.workload,
                    "shape": f"{width}x{height}",
                    "arrival_s": arrival_s,
                    "reference_cost_ms": reference_cost_ms,
                    "slo_ms": slo_ms,
                    "deadline_s": row["deadline_time_s"],
                    "step_cost_ms": estimate.step_ms,
                    "cost_source": estimate.source,
                }
            )

    metadata_path = output.with_suffix(output.suffix + ".manifest.json")
    metadata_path.write_text(json.dumps(metadata_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_profile_rows(policy_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(policy_dir.glob("step_cost_raw*.jsonl")):
        rows.extend(_read_jsonl(path))
    return rows


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo))


def _scale_dir(output_dir: Path, policy: str, scale: float) -> Path:
    candidates = [
        output_dir / policy / f"scale_{scale:g}",
        output_dir / policy / f"scale_{scale:.1f}",
        output_dir / policy / f"scale_{scale}",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _format_dist(counter: dict[int, int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _bucket_stats(raw_rows: list[dict[str, Any]], policy: str, scale: float) -> dict[str, Any]:
    rows = []
    for row in raw_rows:
        tags = row.get("profile_tags") or {}
        if tags.get("profile_mode") != "e2e_slo_first":
            continue
        if tags.get("profile_policy") != policy:
            continue
        try:
            row_scale = float(tags.get("profile_scale"))
        except (TypeError, ValueError):
            continue
        if abs(row_scale - scale) > 1e-9:
            continue
        if row.get("interrupted"):
            continue
        if row.get("batch_size") is None:
            continue
        rows.append(row)

    batch_sizes = [float(row.get("batch_size") or 0.0) for row in rows]
    batch_counter = Counter(int(size) for size in batch_sizes)
    denoise_ms = [float(row.get("denoise_ms") or 0.0) for row in rows if row.get("denoise_ms") is not None]
    by_shape: dict[str, list[float]] = defaultdict(list)
    by_replica: dict[str, int] = defaultdict(int)
    for row in rows:
        by_shape[str(row.get("shape_key"))].append(float(row.get("batch_size") or 0.0))
        by_replica[str(row.get("replica_id"))] += 1

    return {
        "step_ticks": len(rows),
        "profile_rows_for_scale": len(rows),
        "raw_profile_rows": len(rows),
        "raw_profile_rows_policy_total": len(raw_rows),
        "mean_bucket_size": statistics.mean(batch_sizes) if batch_sizes else 0.0,
        "p95_bucket_size": _percentile(batch_sizes, 95),
        "max_bucket_size": max(batch_sizes) if batch_sizes else 0.0,
        "bucket_size_distribution": _format_dist(dict(batch_counter)),
        "mean_denoise_ms": statistics.mean(denoise_ms) if denoise_ms else 0.0,
        "p95_denoise_ms": _percentile(denoise_ms, 95),
        "mean_bucket_size_by_shape": {
            shape: statistics.mean(values) if values else 0.0 for shape, values in sorted(by_shape.items())
        },
        "step_ticks_by_replica": dict(sorted(by_replica.items())),
    }


def _miss_by_shape(requests: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for req in requests:
        shape = f"{req.get('width')}x{req.get('height')}"
        stats = out.setdefault(shape, {"requests": 0, "misses": 0})
        stats["requests"] += 1
        stats["misses"] += 0 if req.get("slo_achieved") else 1
    return out


def _latency_by_shape(requests: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for req in requests:
        if not req.get("success"):
            continue
        shape = f"{req.get('width')}x{req.get('height')}"
        grouped[shape].append(float(req.get("latency_s") or 0.0))
    return {
        shape: {
            "count": len(values),
            "mean_s": statistics.mean(values) if values else 0.0,
            "p95_s": _percentile(values, 95),
        }
        for shape, values in sorted(grouped.items())
    }


def summarize(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    scales = [float(item.strip()) for item in args.scales.split(",") if item.strip()]

    rows: list[dict[str, Any]] = []
    for policy in policies:
        raw_rows = _read_profile_rows(output_dir / policy)
        for scale in scales:
            result_path = _scale_dir(output_dir, policy, scale) / "benchmark_result.json"
            if not result_path.exists():
                rows.append({"policy": policy, "slo_scale": scale, "status": "missing"})
                continue
            payload = _read_json(result_path)
            metrics = payload.get("metrics") or {}
            requests = payload.get("requests") or []
            completed = int(metrics.get("completed_requests") or 0)
            failed = int(metrics.get("failed_requests") or 0)
            slo_met = int(metrics.get("slo_met_success") or 0)
            miss_rate = 1.0 - (float(metrics.get("slo_attainment_rate") or 0.0))
            duration = float(metrics.get("duration") or 0.0)
            bucket = _bucket_stats(raw_rows, policy, scale)
            rows.append(
                {
                    "policy": policy,
                    "slo_scale": scale,
                    "status": "ok",
                    "completed": completed,
                    "failed": failed,
                    "slo_met": slo_met,
                    "slo_miss_rate": miss_rate,
                    "goodput_rps": (slo_met / duration) if duration > 0 else 0.0,
                    "throughput_rps": float(metrics.get("throughput_qps") or 0.0),
                    "latency_p50_s": float(metrics.get("latency_p50") or 0.0),
                    "latency_p95_s": float(metrics.get("latency_p95") or 0.0),
                    "latency_p99_s": float(metrics.get("latency_p99") or 0.0),
                    "misses_by_shape": _miss_by_shape(requests),
                    "latency_by_shape": _latency_by_shape(requests),
                    **bucket,
                }
            )

    summary = {"output_dir": str(output_dir), "rows": rows}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# E2E SLO First Experiment",
        "",
        "| Policy | SLO scale | Miss rate | Goodput | Throughput | Avg bucket | Step ticks | P95 latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(f"| {row['policy']} | {row['slo_scale']} | missing | | | | | |")
            continue
        lines.append(
            "| {policy} | {scale:g} | {miss:.2%} | {goodput:.4f} | {throughput:.4f} | "
            "{bucket:.3f} | {ticks} | {p95:.2f}s |".format(
                policy=row["policy"],
                scale=float(row["slo_scale"]),
                miss=float(row["slo_miss_rate"]),
                goodput=float(row["goodput_rps"]),
                throughput=float(row["throughput_rps"]),
                bucket=float(row["mean_bucket_size"]),
                ticks=int(row["step_ticks"]),
                p95=float(row["latency_p95_s"]),
            )
        )
    lines.append("")
    lines.extend(
        [
            "",
            "## Bucket Distribution",
            "",
            "| Policy | SLO scale | Bucket size distribution | Mean bucket by shape |",
            "|---|---:|---|---|",
        ]
    )
    for row in rows:
        if row.get("status") != "ok":
            continue
        lines.append(
            "| {policy} | {scale:g} | {dist} | {shape} |".format(
                policy=row["policy"],
                scale=float(row["slo_scale"]),
                dist=json.dumps(row["bucket_size_distribution"], ensure_ascii=False, sort_keys=True),
                shape=json.dumps(row["mean_bucket_size_by_shape"], ensure_ascii=False, sort_keys=True),
            )
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and summarize the first Qwen-Image e2e SLO experiment.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    make = subparsers.add_parser("make-trace")
    make.add_argument("--output", required=True)
    make.add_argument("--trace-id", required=True)
    make.add_argument("--profile-policy", default="")
    make.add_argument("--profile-mode", default="e2e_slo_first")
    make.add_argument("--profile-repeat", type=int, default=0)
    make.add_argument("--workload", default="current-mix", choices=sorted(WORKLOAD_SHAPES))
    make.add_argument("--num-requests", type=int, default=80)
    make.add_argument("--global-interarrival-s", type=float, default=4.25)
    make.add_argument("--burst-size", type=int, default=8)
    make.add_argument("--burst-interarrival-s", type=float, default=0.35)
    make.add_argument("--slo-scale", type=float, required=True)
    make.add_argument("--num-inference-steps", type=int, default=50)
    make.add_argument("--seed", type=int, default=1234)
    make.add_argument("--model", default=MODEL)
    make.add_argument("--imagenet-readme", default="/dataset/datasets/imagenet-1k/README.md")
    make.add_argument("--step-cost-model-path", default=DEFAULT_COST_MODEL)
    make.add_argument("--step-cost-metric", default="p90_ms")
    make.add_argument("--step-cost-formula", default="qwen_image_910b_tp2_v1")
    make.set_defaults(func=make_trace)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--output-dir", required=True)
    summary.add_argument("--policies", default="current,full_slo")
    summary.add_argument("--scales", default="2.5,4.0,5.0")
    summary.set_defaults(func=summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
