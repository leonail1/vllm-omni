# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
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


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _scale_label(scale: float) -> str:
    return f"{scale:g}"


def _scale_dir(policy_dir: Path, repeat: int, scale: float) -> Path:
    repeat_dir = policy_dir / f"repeat_{repeat}"
    candidates = [
        repeat_dir / f"scale_{scale:g}",
        repeat_dir / f"scale_{scale:.1f}",
        repeat_dir / f"scale_{scale}",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _filtered_raw_rows(
    policy_dir: Path,
    *,
    profile_mode: str,
    policy: str,
    repeat: int,
    scale: float,
) -> tuple[list[dict[str, Any]], int]:
    all_rows: list[dict[str, Any]] = []
    for path in sorted(policy_dir.glob("step_cost_raw*.jsonl")):
        all_rows.extend(_read_jsonl(path))

    rows = []
    for row in all_rows:
        tags = row.get("profile_tags") or {}
        if tags.get("profile_mode") != profile_mode:
            continue
        if tags.get("profile_policy") != policy:
            continue
        try:
            row_scale = float(tags.get("profile_scale"))
            row_repeat = int(tags.get("profile_repeat"))
        except (TypeError, ValueError):
            continue
        if abs(row_scale - scale) > 1e-9 or row_repeat != repeat:
            continue
        if row.get("interrupted"):
            continue
        rows.append(row)
    return rows, len(all_rows)


def _flatten_numbers(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, list):
            for item in value:
                if item is not None:
                    out.append(float(item))
        elif value is not None:
            out.append(float(value))
    return out


def _bucket_stats(rows: list[dict[str, Any]], raw_total: int) -> dict[str, Any]:
    batch_sizes = [float(row.get("batch_size") or 0.0) for row in rows]
    batch_counter = Counter(int(size) for size in batch_sizes)
    denoise_ms = [float(row.get("denoise_ms") or 0.0) for row in rows if row.get("denoise_ms") is not None]
    total_ms = [float(row.get("total_step_ms") or 0.0) for row in rows if row.get("total_step_ms") is not None]
    time_to_deadline_ms = _flatten_numbers(rows, "time_to_deadline_ms")

    by_replica: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_replica[str(row.get("replica_id"))].append(row)

    replica_stats = {}
    for replica, replica_rows in sorted(by_replica.items()):
        timestamps = [float(row.get("timestamp_s") or 0.0) for row in replica_rows]
        busy_ms = sum(float(row.get("total_step_ms") or row.get("denoise_ms") or 0.0) for row in replica_rows)
        if timestamps:
            end_s = max(
                float(row.get("timestamp_s") or 0.0)
                + float(row.get("total_step_ms") or row.get("denoise_ms") or 0.0) / 1000.0
                for row in replica_rows
            )
            window_ms = max((end_s - min(timestamps)) * 1000.0, 0.0)
        else:
            window_ms = 0.0
        replica_stats[replica] = {
            "step_ticks": len(replica_rows),
            "busy_ms": busy_ms,
            "window_ms": window_ms,
            "idle_ms": max(window_ms - busy_ms, 0.0),
            "utilization": busy_ms / window_ms if window_ms > 0 else 0.0,
        }

    return {
        "step_ticks": len(rows),
        "profile_rows_for_run": len(rows),
        "raw_profile_rows_policy_total": raw_total,
        "mean_bucket_size": _mean(batch_sizes),
        "p95_bucket_size": _percentile(batch_sizes, 95),
        "max_bucket_size": max(batch_sizes) if batch_sizes else 0.0,
        "bucket_size_distribution": {str(k): int(v) for k, v in sorted(batch_counter.items())},
        "mean_denoise_ms": _mean(denoise_ms),
        "p95_denoise_ms": _percentile(denoise_ms, 95),
        "mean_total_step_ms": _mean(total_ms),
        "p95_total_step_ms": _percentile(total_ms, 95),
        "p5_time_to_deadline_ms": _percentile(time_to_deadline_ms, 5),
        "mean_time_to_deadline_ms": _mean(time_to_deadline_ms),
        "replica_stats": replica_stats,
    }


def _stagepool_stats(
    policy_dir: Path,
    *,
    response_ids: set[str],
    client_request_ids: set[str],
    trace_ids: list[str],
) -> dict[str, Any]:
    rows = _read_jsonl(policy_dir / "stagepool_profile.jsonl")
    matched_rows = []
    matched_request_keys = set()
    selected = []
    null_selection_rows = 0
    slo_null_selection_rows = 0
    candidates_per_request = []
    events = Counter()
    for row in rows:
        request_id = str(row.get("request_id") or "")
        client_request_id = str(row.get("client_request_id") or "")
        # StagePool 的 request_id 是服务端 id（chatcmpl-...），benchmark 结果里
        # 对应 response_id；失败请求可能没有 response_id，所以新 profile 同时
        # 写 client_request_id 来兜住失败/timeout 样本。旧 profile 继续用 trace 前缀兜底。
        if (
            request_id not in response_ids
            and client_request_id not in client_request_ids
            and not any(
                request_id.startswith(f"{trace_id}-") for trace_id in trace_ids
            )
        ):
            continue
        matched_rows.append(row)
        matched_key = client_request_id or request_id
        if matched_key:
            matched_request_keys.add(matched_key)
        event = str(row.get("event") or "")
        if event:
            events[event] += 1
        selected_replica = row.get("selected_replica_id")
        if selected_replica is not None:
            selected.append(str(selected_replica))
        else:
            null_selection_rows += 1
            if "slo_select" in event:
                slo_null_selection_rows += 1
        candidates = row.get("candidates")
        if isinstance(candidates, list):
            candidates_per_request.append(float(len(candidates)))
    return {
        "stagepool_profile_rows": len(matched_rows),
        "stagepool_profile_rows_total": len(rows),
        "stagepool_policy_file_rows_total": len(rows),
        "stagepool_selected_rows": len(selected),
        "stagepool_matched_request_rows": len(matched_request_keys),
        "stagepool_missing_request_rows": max(len(client_request_ids) - len(matched_request_keys), 0),
        "stagepool_null_selection_rows": null_selection_rows,
        "stagepool_slo_null_selection_rows": slo_null_selection_rows,
        "stagepool_selected_replica_distribution": dict(sorted(Counter(selected).items())),
        "stagepool_event_distribution": dict(sorted(events.items())),
        "stagepool_mean_candidates": _mean(candidates_per_request),
    }


def _request_stats(payload: dict[str, Any]) -> dict[str, Any]:
    requests = payload.get("requests") or []
    by_shape: dict[str, dict[str, int]] = {}
    for req in requests:
        shape = f"{req.get('width')}x{req.get('height')}"
        stats = by_shape.setdefault(shape, {"requests": 0, "misses": 0})
        stats["requests"] += 1
        stats["misses"] += 0 if req.get("slo_achieved") else 1
    return {"misses_by_shape": by_shape}


def _summarize_one(
    output_dir: Path,
    *,
    profile_mode: str,
    policy: str,
    repeat: int,
    scale: float,
) -> dict[str, Any]:
    policy_dir = output_dir / policy
    result_path = _scale_dir(policy_dir, repeat, scale) / "benchmark_result.json"
    if not result_path.exists():
        return {"policy": policy, "repeat": repeat, "slo_scale": scale, "status": "missing"}

    payload = _read_json(result_path)
    metrics = payload.get("metrics") or {}
    rows, raw_total = _filtered_raw_rows(
        policy_dir,
        profile_mode=profile_mode,
        policy=policy,
        repeat=repeat,
        scale=scale,
    )
    trace_ids = list(
        dict.fromkeys(
            [
                f"{policy}_r{repeat}_scale_{_scale_label(scale)}",
                f"{policy}_r{repeat}_scale_{scale:.1f}",
                f"{policy}_r{repeat}_scale_{scale}",
            ]
        )
    )
    completed = int(metrics.get("completed_requests") or 0)
    failed = int(metrics.get("failed_requests") or 0)
    slo_met = int(metrics.get("slo_met_success") or 0)
    duration = float(metrics.get("duration") or 0.0)
    row = {
        "policy": policy,
        "repeat": repeat,
        "slo_scale": scale,
        "status": "ok",
        "completed": completed,
        "failed": failed,
        "slo_met": slo_met,
        "slo_miss_rate": 1.0 - float(metrics.get("slo_attainment_rate") or 0.0),
        "goodput_rps": slo_met / duration if duration > 0 else 0.0,
        "throughput_rps": float(metrics.get("throughput_qps") or 0.0),
        "duration_s": duration,
        "latency_p50_s": float(metrics.get("latency_p50") or 0.0),
        "latency_p95_s": float(metrics.get("latency_p95") or 0.0),
        "latency_p99_s": float(metrics.get("latency_p99") or 0.0),
    }
    row.update(_request_stats(payload))
    row.update(_bucket_stats(rows, raw_total))
    response_ids = {
        str(req.get("response_id"))
        for req in payload.get("requests", [])
        if req.get("response_id")
    }
    client_request_ids = {
        str(req.get("request_id"))
        for req in payload.get("requests", [])
        if req.get("request_id")
    }
    row.update(
        _stagepool_stats(
            policy_dir,
            response_ids=response_ids,
            client_request_ids=client_request_ids,
            trace_ids=trace_ids,
        )
    )
    return row


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[(str(row["policy"]), float(row["slo_scale"]))].append(row)

    out = []
    metrics = [
        "slo_miss_rate",
        "goodput_rps",
        "throughput_rps",
        "latency_p95_s",
        "duration_s",
        "mean_bucket_size",
        "p95_bucket_size",
        "mean_denoise_ms",
        "p95_denoise_ms",
        "p5_time_to_deadline_ms",
        "mean_time_to_deadline_ms",
        "stagepool_profile_rows",
        "stagepool_matched_request_rows",
        "stagepool_selected_rows",
        "stagepool_missing_request_rows",
        "stagepool_null_selection_rows",
        "stagepool_slo_null_selection_rows",
    ]
    for (policy, scale), items in sorted(grouped.items()):
        row: dict[str, Any] = {"policy": policy, "slo_scale": scale, "repeats": len(items)}
        for metric in metrics:
            values = [float(item.get(metric) or 0.0) for item in items]
            row[f"{metric}_mean"] = _mean(values)
            row[f"{metric}_std"] = _stdev(values)
        out.append(row)
    return out


def summarize(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    scales = [float(item.strip()) for item in args.scales.split(",") if item.strip()]
    repeats = [int(item.strip()) for item in args.repeats.split(",") if item.strip()]

    rows = [
        _summarize_one(
            output_dir,
            profile_mode=args.profile_mode,
            policy=policy,
            repeat=repeat,
            scale=scale,
        )
        for policy in policies
        for repeat in repeats
        for scale in scales
    ]
    aggregate = _aggregate(rows)
    missing_runs = [
        {
            "policy": row.get("policy"),
            "repeat": row.get("repeat"),
            "slo_scale": row.get("slo_scale"),
        }
        for row in rows
        if row.get("status") != "ok"
    ]
    missing_by_policy_scale = Counter(
        (str(row["policy"]), float(row["slo_scale"])) for row in rows if row.get("status") != "ok"
    )
    expected_repeats = len(repeats)
    for row in aggregate:
        row["expected_repeats"] = expected_repeats
        row["missing_runs"] = int(missing_by_policy_scale.get((str(row["policy"]), float(row["slo_scale"])), 0))
    payload = {
        "output_dir": str(output_dir),
        "profile_mode": args.profile_mode,
        "missing_runs": missing_runs,
        "rows": rows,
        "aggregate": aggregate,
    }
    (output_dir / "ablation_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    table_fields = [
        "policy",
        "slo_scale",
        "expected_repeats",
        "repeats",
        "missing_runs",
        "slo_miss_rate_mean",
        "slo_miss_rate_std",
        "goodput_rps_mean",
        "throughput_rps_mean",
        "latency_p95_s_mean",
        "duration_s_mean",
        "mean_bucket_size_mean",
        "p5_time_to_deadline_ms_mean",
        "stagepool_profile_rows_mean",
        "stagepool_missing_request_rows_mean",
    ]
    with (output_dir / "ablation_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=table_fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(aggregate)

    lines = [
        "# E2E SLO Ablation Summary",
        "",
        "| Policy | SLO scale | Repeats | Miss rate | Goodput | Throughput | P95 latency | Mean bucket | P5 TTD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {policy} | {scale:g} | {repeats} | {miss:.2%} | {goodput:.4f} | {throughput:.4f} | "
            "{p95:.2f}s | {bucket:.3f} | {ttd:.1f}ms |".format(
                policy=row["policy"],
                scale=float(row["slo_scale"]),
                repeats=int(row["repeats"]),
                miss=float(row["slo_miss_rate_mean"]),
                goodput=float(row["goodput_rps_mean"]),
                throughput=float(row["throughput_rps_mean"]),
                p95=float(row["latency_p95_s_mean"]),
                bucket=float(row["mean_bucket_size_mean"]),
                ttd=float(row["p5_time_to_deadline_ms_mean"]),
            )
        )
    if missing_runs:
        lines.extend(["", "## Missing Runs", ""])
        for row in missing_runs:
            lines.append(f"- {row['policy']} repeat={row['repeat']} scale={row['slo_scale']}")
    (output_dir / "ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Qwen-Image SLO scheduler ablation results.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policies", required=True)
    parser.add_argument("--scales", default="2.5,4.0")
    parser.add_argument("--repeats", default="0,1,2")
    parser.add_argument("--profile-mode", default="e2e_slo_ablation")
    args = parser.parse_args()
    summarize(args)


if __name__ == "__main__":
    main()
