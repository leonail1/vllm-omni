# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
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


def _interarrival_token(interarrival_s: float) -> str:
    return f"{interarrival_s:g}".replace(".", "p")


def _load_label(interarrival_s: float) -> str:
    return f"ia_{_interarrival_token(interarrival_s)}"


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


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


def _result_policy_dir(output_dir: Path, workload: str, interarrival_s: float, policy: str) -> Path:
    return output_dir / workload / _load_label(interarrival_s) / policy


def _profile_policy_dir(output_dir: Path, policy: str) -> Path:
    return output_dir / policy


def _trace_ids(workload: str, interarrival_s: float, policy: str, repeat: int, scale: float) -> list[str]:
    load_label = _load_label(interarrival_s)
    return list(
        dict.fromkeys(
            [
                f"{workload}_{load_label}_{policy}_r{repeat}_scale_{_scale_label(scale)}",
                f"{workload}_{load_label}_{policy}_r{repeat}_scale_{scale:.1f}",
                f"{workload}_{load_label}_{policy}_r{repeat}_scale_{scale}",
            ]
        )
    )


def _row_request_ids(row: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in ("request_ids", "scheduled_req_ids"):
        value = row.get(key)
        if isinstance(value, list):
            ids.update(str(item) for item in value if item)
        elif value:
            ids.add(str(value))
    return ids


def _tag_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if value is None:
        return []
    return [str(value)]


def _tag_float_values(value: Any) -> list[float]:
    out = []
    for item in _tag_values(value):
        try:
            out.append(float(item))
        except ValueError:
            pass
    return out


def _tags_match_run(
    tags: dict[str, Any],
    *,
    workload: str,
    interarrival_s: float,
    policy: str,
    repeat: int,
    scale: float,
) -> bool:
    trace_values = _tag_values(tags.get("profile_trace_id"))
    if trace_values:
        expected = set(_trace_ids(workload, interarrival_s, policy, repeat, scale))
        return any(trace_id in expected for trace_id in trace_values)

    workload_values = _tag_values(tags.get("profile_workload"))
    if workload_values and workload not in workload_values:
        return False
    load_values = _tag_values(tags.get("profile_load_label"))
    if load_values and _load_label(interarrival_s) not in load_values:
        return False
    interarrival_values = _tag_float_values(tags.get("profile_interarrival_s"))
    if interarrival_values and all(abs(value - interarrival_s) > 1e-9 for value in interarrival_values):
        return False
    return bool(workload_values and (load_values or interarrival_values))


def _filtered_raw_rows(
    policy_dir: Path,
    *,
    profile_mode: str,
    policy: str,
    workload: str,
    interarrival_s: float,
    repeat: int,
    scale: float,
    response_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], int, int, int]:
    all_rows: list[dict[str, Any]] = []
    for path in sorted(policy_dir.glob("step_cost_raw*.jsonl")):
        all_rows.extend(_read_jsonl(path))

    rows = []
    matched_before_response_filter = 0
    dropped_by_response_filter = 0
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
        if not _tags_match_run(
            tags,
            workload=workload,
            interarrival_s=interarrival_s,
            policy=policy,
            repeat=repeat,
            scale=scale,
        ):
            continue
        if row.get("interrupted"):
            continue
        matched_before_response_filter += 1
        if response_ids is not None and not (_row_request_ids(row) & response_ids):
            dropped_by_response_filter += 1
            continue
        rows.append(row)
    return rows, len(all_rows), matched_before_response_filter, dropped_by_response_filter


def _flatten_numbers(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, list):
            out.extend(float(item) for item in value if item is not None)
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
        if (
            request_id not in response_ids
            and client_request_id not in client_request_ids
            and not any(request_id.startswith(f"{trace_id}-") for trace_id in trace_ids)
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
    by_shape: dict[str, dict[str, int]] = {}
    for req in payload.get("requests") or []:
        shape = f"{req.get('width')}x{req.get('height')}"
        stats = by_shape.setdefault(shape, {"requests": 0, "misses": 0})
        stats["requests"] += 1
        stats["misses"] += 0 if req.get("slo_achieved") else 1
    return {"misses_by_shape": by_shape}


def _summarize_one(
    output_dir: Path,
    *,
    workload: str,
    interarrival_s: float,
    profile_mode: str,
    policy: str,
    repeat: int,
    scale: float,
) -> dict[str, Any]:
    result_policy_dir = _result_policy_dir(output_dir, workload, interarrival_s, policy)
    result_path = _scale_dir(result_policy_dir, repeat, scale) / "benchmark_result.json"
    if not result_path.exists():
        return {
            "workload": workload,
            "interarrival_s": interarrival_s,
            "load_label": _load_label(interarrival_s),
            "policy": policy,
            "repeat": repeat,
            "slo_scale": scale,
            "status": "missing",
        }

    payload = _read_json(result_path)
    metrics = payload.get("metrics") or {}
    completed = int(metrics.get("completed_requests") or 0)
    failed = int(metrics.get("failed_requests") or 0)
    slo_met = int(metrics.get("slo_met_success") or 0)
    duration = float(metrics.get("duration") or 0.0)
    response_ids = {
        str(req.get("response_id"))
        for req in payload.get("requests", [])
        if req.get("response_id")
    }
    policy_dir = _profile_policy_dir(output_dir, policy)
    rows, raw_total, profile_rows_before_response_filter, profile_rows_dropped_by_response_filter = _filtered_raw_rows(
        policy_dir,
        profile_mode=profile_mode,
        policy=policy,
        workload=workload,
        interarrival_s=interarrival_s,
        repeat=repeat,
        scale=scale,
        response_ids=response_ids or None,
    )
    trace_ids = _trace_ids(workload, interarrival_s, policy, repeat, scale)
    row = {
        "workload": workload,
        "interarrival_s": interarrival_s,
        "load_label": _load_label(interarrival_s),
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
    row["profile_rows_before_response_filter"] = profile_rows_before_response_filter
    row["profile_rows_dropped_by_response_filter"] = profile_rows_dropped_by_response_filter
    if completed > 0 and row["profile_rows_for_run"] <= 0:
        row["status"] = "profile_missing"
        row["profile_warning"] = (
            "No matching step profile rows found for this completed run; "
            "bucket, deadline, and replica-utilization metrics are not trustworthy."
        )
    elif failed > 0 and profile_rows_dropped_by_response_filter > 0:
        row["profile_warning"] = (
            "Some matching step profile rows were dropped by response_id filtering because "
            "this run has failed requests without response ids; profile-derived metrics may be biased."
        )
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
    grouped: dict[tuple[str, float, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[
                (
                    str(row["workload"]),
                    float(row["interarrival_s"]),
                    str(row["policy"]),
                    float(row["slo_scale"]),
                )
            ].append(row)

    metrics = [
        "slo_miss_rate",
        "goodput_rps",
        "throughput_rps",
        "latency_p95_s",
        "duration_s",
        "mean_bucket_size",
        "p95_bucket_size",
        "profile_rows_before_response_filter",
        "profile_rows_dropped_by_response_filter",
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
    out = []
    for (workload, interarrival_s, policy, scale), items in sorted(grouped.items()):
        row: dict[str, Any] = {
            "workload": workload,
            "interarrival_s": interarrival_s,
            "load_label": _load_label(interarrival_s),
            "policy": policy,
            "slo_scale": scale,
            "repeats": len(items),
        }
        for metric in metrics:
            values = [float(item.get(metric) or 0.0) for item in items]
            row[f"{metric}_mean"] = _mean(values)
            row[f"{metric}_std"] = _stdev(values)
        out.append(row)
    return out


def _metric(row: dict[str, Any] | None, name: str) -> float:
    if row is None:
        return 0.0
    value = row.get(f"{name}_mean", row.get(name, 0.0))
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _comparison_rows(aggregate: list[dict[str, Any]], candidate_policy: str) -> list[dict[str, Any]]:
    by_key = {
        (
            str(row["workload"]),
            float(row["interarrival_s"]),
            float(row["slo_scale"]),
            str(row["policy"]),
        ): row
        for row in aggregate
    }
    keys = sorted({(workload, interarrival, scale) for workload, interarrival, scale, _ in by_key})
    rows = []
    for workload, interarrival_s, scale in keys:
        base = by_key.get((workload, interarrival_s, scale, "current"))
        candidate = by_key.get((workload, interarrival_s, scale, candidate_policy))
        if base is None or candidate is None:
            continue
        base_miss = _metric(base, "slo_miss_rate")
        cand_miss = _metric(candidate, "slo_miss_rate")
        base_goodput = _metric(base, "goodput_rps")
        cand_goodput = _metric(candidate, "goodput_rps")
        base_throughput = _metric(base, "throughput_rps")
        cand_throughput = _metric(candidate, "throughput_rps")
        base_p95 = _metric(base, "latency_p95_s")
        cand_p95 = _metric(candidate, "latency_p95_s")
        rows.append(
            {
                "workload": workload,
                "interarrival_s": interarrival_s,
                "load_label": _load_label(interarrival_s),
                "slo_scale": scale,
                "miss_rate_current": base_miss,
                "miss_rate_candidate": cand_miss,
                "miss_rate_delta_pp": (base_miss - cand_miss) * 100.0,
                "goodput_current": base_goodput,
                "goodput_candidate": cand_goodput,
                "goodput_improvement_pct": ((cand_goodput / base_goodput) - 1.0) * 100.0 if base_goodput > 0 else 0.0,
                "throughput_current": base_throughput,
                "throughput_candidate": cand_throughput,
                "throughput_change_pct": ((cand_throughput / base_throughput) - 1.0) * 100.0 if base_throughput > 0 else 0.0,
                "p95_latency_current_s": base_p95,
                "p95_latency_candidate_s": cand_p95,
                "p95_latency_change_pct": ((cand_p95 / base_p95) - 1.0) * 100.0 if base_p95 > 0 else 0.0,
                "mean_bucket_current": _metric(base, "mean_bucket_size"),
                "mean_bucket_candidate": _metric(candidate, "mean_bucket_size"),
            }
        )
    return rows


def _frontier_rows(aggregate: list[dict[str, Any]], thresholds: list[float]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate:
        grouped[(str(row["workload"]), str(row["policy"]), float(row["slo_scale"]))].append(row)

    out = []
    for workload, policy, scale in sorted(grouped):
        candidates = grouped[(workload, policy, scale)]
        for threshold in thresholds:
            eligible = [
                row for row in candidates
                if _metric(row, "slo_miss_rate") <= threshold
            ]
            if not eligible:
                out.append(
                    {
                        "workload": workload,
                        "policy": policy,
                        "slo_scale": scale,
                        "miss_threshold": threshold,
                        "status": "no_eligible_run",
                    }
                )
                continue
            selected = max(
                eligible,
                key=lambda row: (
                    _metric(row, "throughput_rps"),
                    _metric(row, "goodput_rps"),
                    -float(row["interarrival_s"]),
                ),
            )
            out.append(
                {
                    "workload": workload,
                    "policy": policy,
                    "slo_scale": scale,
                    "miss_threshold": threshold,
                    "status": "ok",
                    "selected_interarrival_s": float(selected["interarrival_s"]),
                    "selected_load_label": selected["load_label"],
                    "throughput_rps": _metric(selected, "throughput_rps"),
                    "goodput_rps": _metric(selected, "goodput_rps"),
                    "slo_miss_rate": _metric(selected, "slo_miss_rate"),
                    "latency_p95_s": _metric(selected, "latency_p95_s"),
                    "mean_bucket_size": _metric(selected, "mean_bucket_size"),
                    "p5_time_to_deadline_ms": _metric(selected, "p5_time_to_deadline_ms"),
                }
            )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summary_table_fields() -> list[str]:
    return [
        "workload",
        "interarrival_s",
        "load_label",
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
        "p95_bucket_size_mean",
        "p5_time_to_deadline_ms_mean",
        "mean_time_to_deadline_ms_mean",
        "stagepool_profile_rows_mean",
        "stagepool_missing_request_rows_mean",
        "profile_rows_before_response_filter_mean",
        "profile_rows_dropped_by_response_filter_mean",
    ]


def _comparison_fields() -> list[str]:
    return [
        "workload",
        "interarrival_s",
        "load_label",
        "slo_scale",
        "miss_rate_current",
        "miss_rate_candidate",
        "miss_rate_delta_pp",
        "goodput_current",
        "goodput_candidate",
        "goodput_improvement_pct",
        "throughput_current",
        "throughput_candidate",
        "throughput_change_pct",
        "p95_latency_current_s",
        "p95_latency_candidate_s",
        "p95_latency_change_pct",
        "mean_bucket_current",
        "mean_bucket_candidate",
    ]


def _comparison_file_name(candidate_policy: str) -> str:
    safe_policy = candidate_policy.replace("/", "_")
    return f"current_vs_{safe_policy}.csv"


def _frontier_fields() -> list[str]:
    return [
        "workload",
        "policy",
        "slo_scale",
        "miss_threshold",
        "status",
        "selected_interarrival_s",
        "selected_load_label",
        "throughput_rps",
        "goodput_rps",
        "slo_miss_rate",
        "latency_p95_s",
        "mean_bucket_size",
        "p5_time_to_deadline_ms",
    ]


def summarize(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workloads = _parse_csv_strings(args.workloads)
    policies = _parse_csv_strings(args.policies)
    scales = _parse_csv_floats(args.scales)
    repeats = _parse_csv_ints(args.repeats)
    interarrivals = _parse_csv_floats(args.interarrivals)
    thresholds = _parse_csv_floats(args.miss_thresholds)

    rows = [
        _summarize_one(
            output_dir,
            workload=workload,
            interarrival_s=interarrival_s,
            profile_mode=args.profile_mode,
            policy=policy,
            repeat=repeat,
            scale=scale,
        )
        for workload in workloads
        for interarrival_s in interarrivals
        for policy in policies
        for repeat in repeats
        for scale in scales
    ]
    aggregate = _aggregate(rows)
    missing_runs = [
        {
            "workload": row.get("workload"),
            "interarrival_s": row.get("interarrival_s"),
            "load_label": row.get("load_label"),
            "policy": row.get("policy"),
            "repeat": row.get("repeat"),
            "slo_scale": row.get("slo_scale"),
            "status": row.get("status"),
        }
        for row in rows
        if row.get("status") != "ok"
    ]
    missing_by_key = Counter(
        (
            str(row.get("workload")),
            float(row.get("interarrival_s") or 0.0),
            str(row.get("policy")),
            float(row.get("slo_scale") or 0.0),
        )
        for row in rows
        if row.get("status") != "ok"
    )
    for row in aggregate:
        row["expected_repeats"] = len(repeats)
        row["missing_runs"] = int(
            missing_by_key.get(
                (
                    str(row["workload"]),
                    float(row["interarrival_s"]),
                    str(row["policy"]),
                    float(row["slo_scale"]),
                ),
                0,
            )
        )

    comparison = _comparison_rows(aggregate, args.candidate_policy)
    frontier = _frontier_rows(aggregate, thresholds)
    comparison_file_name = _comparison_file_name(args.candidate_policy)
    payload = {
        "output_dir": str(output_dir),
        "profile_mode": args.profile_mode,
        "candidate_policy": args.candidate_policy,
        "comparison_file": comparison_file_name,
        "matrix": {
            "workloads": workloads,
            "interarrivals": interarrivals,
            "policies": policies,
            "scales": scales,
            "repeats": repeats,
            "miss_thresholds": thresholds,
        },
        "missing_runs": missing_runs,
        "rows": rows,
        "aggregate": aggregate,
        "comparison": comparison,
        "frontier": frontier,
    }
    (output_dir / "frontier_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "frontier_table.csv", aggregate, _summary_table_fields())
    _write_csv(output_dir / comparison_file_name, comparison, _comparison_fields())
    _write_csv(output_dir / "frontier_by_threshold.csv", frontier, _frontier_fields())
    _write_report(output_dir, aggregate, comparison, frontier, missing_runs, candidate_policy=args.candidate_policy)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _write_report(
    output_dir: Path,
    aggregate: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    missing_runs: list[dict[str, Any]],
    *,
    candidate_policy: str,
) -> None:
    lines = [
        "# E2E SLO Throughput Frontier",
        "",
        "| Workload | Interarrival | Policy | SLO scale | Repeats | Miss rate | Goodput | Throughput | P95 latency | Mean bucket | P5 TTD |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {workload} | {interarrival:.2f}s | {policy} | {scale:g} | {repeats} | {miss:.2%} | "
            "{goodput:.4f} | {throughput:.4f} | {p95:.2f}s | {bucket:.3f} | {ttd:.1f}ms |".format(
                workload=row["workload"],
                interarrival=float(row["interarrival_s"]),
                policy=row["policy"],
                scale=float(row["slo_scale"]),
                repeats=int(row["repeats"]),
                miss=_metric(row, "slo_miss_rate"),
                goodput=_metric(row, "goodput_rps"),
                throughput=_metric(row, "throughput_rps"),
                p95=_metric(row, "latency_p95_s"),
                bucket=_metric(row, "mean_bucket_size"),
                ttd=_metric(row, "p5_time_to_deadline_ms"),
            )
        )

    lines.extend(
        [
            "",
            f"## Current vs {candidate_policy}",
            "",
            "| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison:
        bucket_delta = float(row["mean_bucket_candidate"]) - float(row["mean_bucket_current"])
        lines.append(
            "| {workload} | {interarrival:.2f}s | {scale:g} | {miss:.2f} pp | {goodput:.2f}% | "
            "{throughput:.2f}% | {p95:.2f}% | {bucket:+.3f} |".format(
                workload=row["workload"],
                interarrival=float(row["interarrival_s"]),
                scale=float(row["slo_scale"]),
                miss=float(row["miss_rate_delta_pp"]),
                goodput=float(row["goodput_improvement_pct"]),
                throughput=float(row["throughput_change_pct"]),
                p95=float(row["p95_latency_change_pct"]),
                bucket=bucket_delta,
            )
        )

    lines.extend(
        [
            "",
            "## Frontier by Miss Threshold",
            "",
            "| Workload | Policy | SLO scale | Miss threshold | Selected interarrival | Throughput | Goodput | Miss rate | P95 latency | Mean bucket |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in frontier:
        if row.get("status") != "ok":
            lines.append(
                "| {workload} | {policy} | {scale:g} | {threshold:.0%} | none | | | | | |".format(
                    workload=row["workload"],
                    policy=row["policy"],
                    scale=float(row["slo_scale"]),
                    threshold=float(row["miss_threshold"]),
                )
            )
            continue
        lines.append(
            "| {workload} | {policy} | {scale:g} | {threshold:.0%} | {interarrival:.2f}s | "
            "{throughput:.4f} | {goodput:.4f} | {miss:.2%} | {p95:.2f}s | {bucket:.3f} |".format(
                workload=row["workload"],
                policy=row["policy"],
                scale=float(row["slo_scale"]),
                threshold=float(row["miss_threshold"]),
                interarrival=float(row["selected_interarrival_s"]),
                throughput=float(row["throughput_rps"]),
                goodput=float(row["goodput_rps"]),
                miss=float(row["slo_miss_rate"]),
                p95=float(row["latency_p95_s"]),
                bucket=float(row["mean_bucket_size"]),
            )
        )

    if missing_runs:
        lines.extend(["", "## Missing or Unusable Runs", ""])
        for row in missing_runs:
            lines.append(
                "- {workload} / {load_label} / {policy} repeat={repeat} scale={scale} status={status}".format(
                    workload=row["workload"],
                    load_label=row["load_label"],
                    policy=row["policy"],
                    repeat=row["repeat"],
                    scale=row["slo_scale"],
                    status=row["status"],
                )
            )
    (output_dir / "frontier_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _shape_rows(
    output_dir: Path,
    *,
    workloads: list[str],
    interarrivals: list[float],
    policies: list[str],
    scales: list[float],
    repeats: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workload in workloads:
        for interarrival_s in interarrivals:
            for policy in policies:
                policy_dir = _result_policy_dir(output_dir, workload, interarrival_s, policy)
                for repeat in repeats:
                    for scale in scales:
                        result_path = _scale_dir(policy_dir, repeat, scale) / "benchmark_result.json"
                        if not result_path.exists():
                            continue
                        payload = _read_json(result_path)
                        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
                        for req in payload.get("requests") or []:
                            grouped[f"{req.get('width')}x{req.get('height')}"].append(req)
                        for shape, requests in sorted(grouped.items()):
                            latencies = [
                                float(req.get("latency_s") or 0.0)
                                for req in requests
                                if req.get("success")
                            ]
                            misses = sum(0 if req.get("slo_achieved") else 1 for req in requests)
                            rows.append(
                                {
                                    "workload": workload,
                                    "interarrival_s": interarrival_s,
                                    "load_label": _load_label(interarrival_s),
                                    "policy": policy,
                                    "repeat": repeat,
                                    "slo_scale": scale,
                                    "shape": shape,
                                    "requests": len(requests),
                                    "misses": misses,
                                    "miss_rate": misses / len(requests) if requests else 0.0,
                                    "latency_mean_s": _mean(latencies),
                                    "latency_p95_s": _percentile(latencies, 95),
                                }
                            )
    return rows


def _distribution_rows(summary: dict[str, Any], *, key: str, value_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary.get("rows") or []:
        if row.get("status") != "ok":
            continue
        dist = row.get(key)
        if not isinstance(dist, dict):
            continue
        for bucket, value in sorted(dist.items(), key=lambda item: str(item[0])):
            rows.append(
                {
                    "workload": row.get("workload"),
                    "interarrival_s": row.get("interarrival_s"),
                    "load_label": row.get("load_label"),
                    "policy": row.get("policy"),
                    "repeat": row.get("repeat"),
                    "slo_scale": row.get("slo_scale"),
                    value_name: bucket,
                    "count": value,
                }
            )
    return rows


def _copy_small_artifacts(
    output_dir: Path,
    bundle_dir: Path,
    *,
    workloads: list[str],
    interarrivals: list[float],
    policies: list[str],
    scales: list[float],
    repeats: list[int],
) -> None:
    for workload in workloads:
        for interarrival_s in interarrivals:
            load_label = _load_label(interarrival_s)
            for policy in policies:
                policy_dir = _result_policy_dir(output_dir, workload, interarrival_s, policy)
                for repeat in repeats:
                    for scale in scales:
                        src_dir = _scale_dir(policy_dir, repeat, scale)
                        if not src_dir.exists():
                            continue
                        dst_dir = (
                            bundle_dir
                            / "benchmark_results"
                            / workload
                            / load_label
                            / policy
                            / f"repeat_{repeat}"
                            / f"scale_{_scale_label(scale)}"
                        )
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        result = src_dir / "benchmark_result.json"
                        if result.exists():
                            shutil.copy2(result, dst_dir / "benchmark_result.json")
                        manifest = src_dir / "trace.txt.manifest.json"
                        if manifest.exists():
                            manifest_dst = (
                                bundle_dir
                                / "trace_manifests"
                                / workload
                                / load_label
                                / policy
                                / f"repeat_{repeat}"
                                / f"scale_{_scale_label(scale)}"
                            )
                            manifest_dst.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(manifest, manifest_dst / "trace.txt.manifest.json")


def _write_omitted_manifest(output_dir: Path, bundle_dir: Path) -> None:
    rows = []
    suffixes = {".jsonl", ".log", ".pid"}
    names = {"trace.txt", "runner.log", "server.log", "client.log"}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in suffixes and path.name not in names:
            continue
        try:
            line_count = sum(1 for _ in path.open("rb"))
        except OSError:
            line_count = 0
        rows.append(
            {
                "relative_path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "lines": line_count,
                "reason": "omitted_large_or_runtime_artifact",
            }
        )
    _write_csv(bundle_dir / "omitted_artifacts_manifest.csv", rows, ["relative_path", "bytes", "lines", "reason"])


def _write_svg(
    path: Path,
    aggregate: list[dict[str, Any]],
    *,
    metric: str,
    title: str,
    as_percent: bool = False,
) -> None:
    rows = sorted(
        aggregate,
        key=lambda row: (
            str(row.get("workload") or ""),
            float(row.get("slo_scale") or 0.0),
            float(row.get("interarrival_s") or 0.0),
            str(row.get("policy") or ""),
        ),
    )
    groups = sorted(
        {
            (str(row.get("workload") or "default"), float(row.get("slo_scale") or 0.0), float(row.get("interarrival_s") or 0.0))
            for row in rows
        }
    )
    policies = sorted({str(row.get("policy") or "") for row in rows})
    if not groups or not policies:
        return

    width = max(1120, 130 + len(groups) * max(64, 26 * len(policies)))
    height = 560
    left = 72
    right = 24
    top = 64
    bottom = 136
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [_metric(row, metric) for row in rows]
    max_value = max(values) if values else 0.0
    if as_percent:
        max_value = max(max_value, 0.01)
    max_value = max_value * 1.15 if max_value > 0 else 1.0
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#f59e0b"]
    by_key = {
        (
            str(row.get("workload") or "default"),
            float(row.get("slo_scale") or 0.0),
            float(row.get("interarrival_s") or 0.0),
            str(row.get("policy") or ""),
        ): row
        for row in rows
    }
    group_w = plot_w / len(groups)
    bar_gap = 4
    bar_w = max((group_w - 16 - bar_gap * (len(policies) - 1)) / len(policies), 7)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
    ]
    for tick in range(6):
        value = max_value * tick / 5
        y = top + plot_h - plot_h * tick / 5
        label = f"{value:.0%}" if as_percent else f"{value:.2f}"
        parts.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{label}</text>')
    for group_idx, (workload, scale, interarrival_s) in enumerate(groups):
        group_x = left + group_idx * group_w + 8
        label_x = group_x + group_w / 2 - 8
        parts.append(
            f'<text x="{label_x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="10">{html.escape(workload)}</text>'
        )
        parts.append(
            f'<text x="{label_x:.1f}" y="{top + plot_h + 38}" text-anchor="middle" font-family="Arial" font-size="10">scale {scale:g}</text>'
        )
        parts.append(
            f'<text x="{label_x:.1f}" y="{top + plot_h + 54}" text-anchor="middle" font-family="Arial" font-size="10">{interarrival_s:g}s</text>'
        )
        for policy_idx, policy in enumerate(policies):
            row = by_key.get((workload, scale, interarrival_s, policy))
            value = _metric(row, metric)
            bar_h = plot_h * value / max_value if max_value > 0 else 0.0
            x = group_x + policy_idx * (bar_w + bar_gap)
            y = top + plot_h - bar_h
            color = colors[policy_idx % len(colors)]
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
    legend_x = left
    legend_y = height - 36
    for policy_idx, policy in enumerate(policies):
        x = legend_x + policy_idx * 250
        color = colors[policy_idx % len(colors)]
        parts.append(f'<rect x="{x}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{x + 18}" y="{legend_y}" font-family="Arial" font-size="12">{html.escape(policy)}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _safe_clean_bundle_dir(bundle_dir: Path) -> None:
    resolved = bundle_dir.resolve()
    if resolved.parent.name != "profile_results" or not resolved.name.startswith("e2e_slo"):
        raise ValueError(
            "--clean is only allowed for dedicated e2e_slo* bundles under "
            f"profile_results, got: {bundle_dir}"
        )


def _scale_labels(scales: list[float]) -> list[str]:
    return [f"{scale:g}" for scale in scales]


def _validate_summary_matrix(
    summary: dict[str, Any],
    *,
    workloads: list[str],
    interarrivals: list[float],
    policies: list[str],
    scales: list[float],
    repeats: list[int],
) -> None:
    matrix = summary.get("matrix")
    if not isinstance(matrix, dict):
        raise ValueError("frontier_summary.json is missing matrix metadata; rerun the summarizer")

    expected = {
        "workloads": workloads,
        "interarrivals": _scale_labels(interarrivals),
        "policies": policies,
        "scales": _scale_labels(scales),
        "repeats": [str(repeat) for repeat in repeats],
    }
    actual = {
        "workloads": [str(item) for item in matrix.get("workloads") or []],
        "interarrivals": _scale_labels([float(item) for item in matrix.get("interarrivals") or []]),
        "policies": [str(item) for item in matrix.get("policies") or []],
        "scales": _scale_labels([float(item) for item in matrix.get("scales") or []]),
        "repeats": [str(item) for item in matrix.get("repeats") or []],
    }
    if actual != expected:
        raise ValueError(f"summary matrix does not match package arguments: actual={actual}, expected={expected}")


def _write_readme(
    bundle_dir: Path,
    *,
    workloads: list[str],
    interarrivals: list[float],
    scales: list[float],
    repeats: list[int],
    policies: list[str],
    candidate_policy: str,
    comparison: list[dict[str, Any]],
) -> None:
    comparison_file_name = _comparison_file_name(candidate_policy)
    lines = [
        "# E2E SLO Throughput Frontier",
        "",
        "This directory contains a lightweight, GitHub-safe result bundle for the no-preemption DiT SLO scheduler throughput-frontier experiment. Runtime logs, raw step profiles, stagepool profiles, and trace text files are intentionally omitted and listed in `omitted_artifacts_manifest.csv`.",
        "",
        "The experiment varies offered load (`interarrival_s`) and SLO scale to see whether the scheduler can keep miss rate low while batching more same-shape work.",
        "",
        "## Experiment Matrix",
        "",
        f"- Workloads: `{', '.join(workloads)}`",
        f"- Interarrival seconds: `{', '.join(_scale_label(item) for item in interarrivals)}`",
        f"- SLO scales: `{', '.join(_scale_label(scale) for scale in scales)}`",
        f"- Repeats: `{', '.join(str(repeat) for repeat in repeats)}`",
        f"- Policies: `{', '.join(policies)}`",
        "",
        f"## Current vs {candidate_policy}",
        "",
        "| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            "| {workload} | {interarrival:.2f}s | {scale:g} | {miss:.2f} pp | {goodput:.2f}% | {throughput:.2f}% | {p95:.2f}% | {bucket:+.3f} |".format(
                workload=row["workload"],
                interarrival=float(row["interarrival_s"]),
                scale=float(row["slo_scale"]),
                miss=float(row["miss_rate_delta_pp"]),
                goodput=float(row["goodput_improvement_pct"]),
                throughput=float(row["throughput_change_pct"]),
                p95=float(row["p95_latency_change_pct"]),
                bucket=float(row["mean_bucket_candidate"]) - float(row["mean_bucket_current"]),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `frontier_summary.json`: structured summary from the raw output directory.",
            "- `frontier_table.csv`: aggregate policy x workload x load x scale table.",
            "- `frontier_by_threshold.csv`: best throughput under configured miss-rate thresholds.",
            f"- `{comparison_file_name}`: relative deltas against `current`.",
            "- `miss_by_shape.csv`: shape-level miss rate and latency.",
            "- `bucket_size_distribution.csv`: denoise bucket size counts from step profiles.",
            "- `stagepool_replica_distribution.csv`: selected replica counts from StagePool profiles.",
            "- `figures/*.svg`: compact visualizations for miss rate, goodput, throughput, P95 latency, and mean bucket size.",
            "- `benchmark_results/`: small `benchmark_result.json` files only.",
            "- `trace_manifests/`: trace manifests without the full trace text.",
        ]
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    bundle_dir = Path(args.bundle_dir)
    summary_path = output_dir / "frontier_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary: {summary_path}")

    if bundle_dir.exists() and args.clean:
        _safe_clean_bundle_dir(bundle_dir)
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    workloads = _parse_csv_strings(args.workloads)
    interarrivals = _parse_csv_floats(args.interarrivals)
    policies = _parse_csv_strings(args.policies)
    scales = _parse_csv_floats(args.scales)
    repeats = _parse_csv_ints(args.repeats)
    summary = _read_json(summary_path)
    _validate_summary_matrix(
        summary,
        workloads=workloads,
        interarrivals=interarrivals,
        policies=policies,
        scales=scales,
        repeats=repeats,
    )
    aggregate = list(summary.get("aggregate") or [])
    comparison = list(summary.get("comparison") or [])
    frontier = list(summary.get("frontier") or [])
    candidate_policy = str(summary.get("candidate_policy") or args.candidate_policy)
    comparison_file_name = str(summary.get("comparison_file") or _comparison_file_name(candidate_policy))

    shutil.copy2(summary_path, bundle_dir / "frontier_summary.json")
    for name in (
        "frontier_report.md",
        "frontier_table.csv",
        "frontier_by_threshold.csv",
        comparison_file_name,
    ):
        src = output_dir / name
        if src.exists():
            shutil.copy2(src, bundle_dir / name)

    _write_csv(bundle_dir / "results_by_workload_load_policy_scale.csv", aggregate, _summary_table_fields())
    _write_csv(bundle_dir / "frontier_by_threshold.csv", frontier, _frontier_fields())
    _write_csv(
        bundle_dir / "miss_by_shape.csv",
        _shape_rows(
            output_dir,
            workloads=workloads,
            interarrivals=interarrivals,
            policies=policies,
            scales=scales,
            repeats=repeats,
        ),
        [
            "workload",
            "interarrival_s",
            "load_label",
            "policy",
            "repeat",
            "slo_scale",
            "shape",
            "requests",
            "misses",
            "miss_rate",
            "latency_mean_s",
            "latency_p95_s",
        ],
    )
    _write_csv(
        bundle_dir / "bucket_size_distribution.csv",
        _distribution_rows(summary, key="bucket_size_distribution", value_name="bucket_size"),
        ["workload", "interarrival_s", "load_label", "policy", "repeat", "slo_scale", "bucket_size", "count"],
    )
    _write_csv(
        bundle_dir / "stagepool_replica_distribution.csv",
        _distribution_rows(summary, key="stagepool_selected_replica_distribution", value_name="replica_id"),
        ["workload", "interarrival_s", "load_label", "policy", "repeat", "slo_scale", "replica_id", "count"],
    )
    _copy_small_artifacts(
        output_dir,
        bundle_dir,
        workloads=workloads,
        interarrivals=interarrivals,
        policies=policies,
        scales=scales,
        repeats=repeats,
    )
    _write_omitted_manifest(output_dir, bundle_dir)

    figures = bundle_dir / "figures"
    _write_svg(figures / "miss_rate.svg", aggregate, metric="slo_miss_rate", title="SLO Miss Rate", as_percent=True)
    _write_svg(figures / "goodput.svg", aggregate, metric="goodput_rps", title="Goodput")
    _write_svg(figures / "throughput.svg", aggregate, metric="throughput_rps", title="Throughput")
    _write_svg(figures / "p95_latency.svg", aggregate, metric="latency_p95_s", title="P95 Latency")
    _write_svg(figures / "mean_bucket_size.svg", aggregate, metric="mean_bucket_size", title="Mean Bucket Size")
    _write_readme(
        bundle_dir,
        workloads=workloads,
        interarrivals=interarrivals,
        scales=scales,
        repeats=repeats,
        policies=policies,
        candidate_policy=candidate_policy,
        comparison=comparison,
    )
    status = {
        "message": "E2E SLO frontier result bundle packaged",
        "output_dir": str(output_dir),
        "bundle_dir": str(bundle_dir),
        "workloads": workloads,
        "interarrivals": interarrivals,
        "policies": policies,
        "candidate_policy": candidate_policy,
        "comparison_file": comparison_file_name,
        "scales": scales,
        "repeats": repeats,
    }
    (bundle_dir / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize or package E2E SLO throughput-frontier results.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--output-dir", required=True)
    summary.add_argument("--policies", default="current,slo_no_preemption_lookup")
    summary.add_argument("--candidate-policy", default="slo_no_preemption_lookup")
    summary.add_argument("--scales", default="4.0,6.0")
    summary.add_argument("--repeats", default="0")
    summary.add_argument("--workloads", default="current-mix,shape-grouped-current-mix")
    summary.add_argument("--interarrivals", default="4.25,2.5")
    summary.add_argument("--miss-thresholds", default="0.05,0.10")
    summary.add_argument("--profile-mode", default="e2e_slo_frontier")
    summary.set_defaults(func=summarize)

    package_parser = subparsers.add_parser("package")
    package_parser.add_argument("--output-dir", required=True)
    package_parser.add_argument("--bundle-dir", required=True)
    package_parser.add_argument("--workloads", default="current-mix,shape-grouped-current-mix")
    package_parser.add_argument("--interarrivals", default="4.25,2.5")
    package_parser.add_argument("--policies", default="current,slo_no_preemption_lookup")
    package_parser.add_argument("--candidate-policy", default="slo_no_preemption_lookup")
    package_parser.add_argument("--scales", default="4.0,6.0")
    package_parser.add_argument("--repeats", default="0")
    package_parser.add_argument("--clean", action="store_true")
    package_parser.set_defaults(func=package)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
