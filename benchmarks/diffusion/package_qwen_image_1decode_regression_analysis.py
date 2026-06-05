# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = q * (len(values) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values[low]
    frac = rank - low
    return values[low] * (1 - frac) + values[high] * frac


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _result_path(run_dir: Path, label: str) -> Path:
    if label == "baseline_7replica":
        return run_dir / "original_7replica_baseline_result.json"
    return run_dir / "smoke_result.json"


def _load_result(matrix_root: Path, label: str) -> dict[str, Any] | None:
    path = _result_path(matrix_root / label, label)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _scenarios(doc: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not doc:
        return []
    scenarios = doc.get("scenarios")
    return scenarios if isinstance(scenarios, list) else []


def _requests(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    requests = scenario.get("requests")
    return requests if isinstance(requests, list) else []


def _steps(scenario: dict[str, Any]) -> int:
    return int(scenario.get("num_inference_steps") or scenario.get("steps") or 0)


def _latency_s(request: dict[str, Any]) -> float:
    for key in ("e2e_latency_s", "latency_s", "wall_duration_s"):
        value = request.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _stage_events(request: dict[str, Any]) -> list[dict[str, Any]]:
    events = request.get("stage_trace")
    return events if isinstance(events, list) else []


def _event(events: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    return next((event for event in events if event.get("stage") == stage), {})


def _manifest_rows(matrix_root: Path) -> list[dict[str, str]]:
    rows = _read_csv(matrix_root / "matrix_manifest.csv")
    if rows:
        return rows
    return []


def _labels(matrix_root: Path) -> list[str]:
    rows = _manifest_rows(matrix_root)
    if rows:
        return [row["label"] for row in rows if row.get("label")]
    return sorted(path.name for path in matrix_root.iterdir() if path.is_dir())


def _scenario_rows(matrix_root: Path, labels: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in labels:
        doc = _load_result(matrix_root, label)
        for scenario_index, scenario in enumerate(_scenarios(doc)):
            requests = [request for request in _requests(scenario) if request.get("status") in (None, "passed")]
            latencies = [_latency_s(request) for request in requests]
            elapsed = float(
                scenario.get("measured_elapsed_s")
                or scenario.get("elapsed_s")
                or doc.get("measured_elapsed_s", 0.0)
                or 0.0
            )
            throughput = float(scenario.get("throughput_img_s") or (len(requests) / elapsed if elapsed > 0 else 0.0))
            failed = len(_requests(scenario)) - len(requests)
            rows.append(
                {
                    "label": label,
                    "scenario_index": scenario_index,
                    "steps": _steps(scenario),
                    "requests": len(requests),
                    "failed_requests": failed,
                    "elapsed_s": f"{elapsed:.6f}",
                    "throughput_img_s": f"{throughput:.6f}",
                    "latency_mean_s": f"{_mean(latencies):.6f}",
                    "latency_p50_s": f"{_percentile(latencies, 0.50):.6f}",
                    "latency_p95_s": f"{_percentile(latencies, 0.95):.6f}",
                }
            )
    return rows


def _stage_timing_rows(matrix_root: Path, labels: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[float]] = {}
    for label in labels:
        doc = _load_result(matrix_root, label)
        for scenario in _scenarios(doc):
            steps = _steps(scenario)
            for request in _requests(scenario):
                if request.get("status") not in (None, "passed"):
                    continue
                for event in _stage_events(request):
                    stage = str(event.get("stage", ""))
                    duration = event.get("duration_s")
                    if not stage or not isinstance(duration, (int, float)):
                        continue
                    grouped.setdefault((label, steps, stage), []).append(float(duration))
    rows: list[dict[str, Any]] = []
    for (label, steps, stage), values in sorted(grouped.items()):
        rows.append(
            {
                "label": label,
                "steps": steps,
                "stage": stage,
                "events": len(values),
                "duration_mean_s": f"{_mean(values):.6f}",
                "duration_p50_s": f"{_percentile(values, 0.50):.6f}",
                "duration_p95_s": f"{_percentile(values, 0.95):.6f}",
            }
        )
    return rows


def _transfer_rows(stage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in stage_rows:
        stage = str(row["stage"])
        if "payload_" not in stage:
            continue
        rows.append(
            {
                "label": row["label"],
                "steps": row["steps"],
                "transfer_stage": stage,
                "events": row["events"],
                "duration_mean_s": row["duration_mean_s"],
                "duration_p50_s": row["duration_p50_s"],
                "duration_p95_s": row["duration_p95_s"],
            }
        )
    return rows


def _decode_queue_rows(matrix_root: Path, labels: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in labels:
        doc = _load_result(matrix_root, label)
        for scenario in _scenarios(doc):
            steps = _steps(scenario)
            by_replica: dict[int, list[dict[str, float]]] = {}
            for request in _requests(scenario):
                if request.get("status") not in (None, "passed"):
                    continue
                events = _stage_events(request)
                durations = request.get("stage_durations") or {}
                denoise = _event(events, "denoise")
                decode = _event(events, "decode")
                if not decode:
                    continue
                replica_id = int(durations.get("stage_2_route_replica_id", decode.get("replica_id", -1)))
                submit_s = durations.get("stage_2_route_submit_s", denoise.get("end_s", decode.get("start_s")))
                queue_s = 0.0
                if isinstance(submit_s, (int, float)) and isinstance(decode.get("start_s"), (int, float)):
                    queue_s = float(decode["start_s"]) - float(submit_s)
                stage2_gen_s = float(durations.get("stage_2_gen_ms") or 0.0) / 1000.0
                by_replica.setdefault(replica_id, []).append(
                    {
                        "queue_s": queue_s,
                        "decode_s": float(decode.get("duration_s") or 0.0),
                        "stage2_gen_s": stage2_gen_s,
                    }
                )
            for replica_id, values in sorted(by_replica.items()):
                queues = [item["queue_s"] for item in values]
                decodes = [item["decode_s"] for item in values]
                stage2 = [item["stage2_gen_s"] for item in values]
                rows.append(
                    {
                        "label": label,
                        "steps": steps,
                        "decode_replica_id": replica_id,
                        "requests": len(values),
                        "submit_to_decode_mean_s": f"{_mean(queues):.6f}",
                        "submit_to_decode_p95_s": f"{_percentile(queues, 0.95):.6f}",
                        "decode_duration_mean_s": f"{_mean(decodes):.6f}",
                        "decode_duration_p95_s": f"{_percentile(decodes, 0.95):.6f}",
                        "stage2_gen_mean_s": f"{_mean(stage2):.6f}",
                        "stage2_gen_p95_s": f"{_percentile(stage2, 0.95):.6f}",
                    }
                )
    return rows


def _dit_completion_rows(matrix_root: Path, labels: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in labels:
        doc = _load_result(matrix_root, label)
        for scenario_index, scenario in enumerate(_scenarios(doc)):
            steps = _steps(scenario)
            scenario_rows: list[dict[str, Any]] = []
            for request in _requests(scenario):
                if request.get("status") not in (None, "passed"):
                    continue
                events = _stage_events(request)
                durations = request.get("stage_durations") or {}
                denoise = _event(events, "denoise")
                decode = _event(events, "decode")
                if not denoise:
                    continue
                submit_s = durations.get("stage_2_route_submit_s", denoise.get("end_s"))
                decode_start_s = decode.get("start_s") if decode else None
                queue_s = 0.0
                if isinstance(submit_s, (int, float)) and isinstance(decode_start_s, (int, float)):
                    queue_s = float(decode_start_s) - float(submit_s)
                scenario_rows.append(
                    {
                        "label": label,
                        "scenario_index": scenario_index,
                        "steps": steps,
                        "request_index": request.get("request_index"),
                        "denoise_replica_id": int(denoise.get("replica_id", -1)),
                        "decode_replica_id": int(durations.get("stage_2_route_replica_id", decode.get("replica_id", -1) if decode else -1)),
                        "denoise_end_wall_s": float(denoise.get("end_s") or 0.0),
                        "decode_start_wall_s": float(decode_start_s or 0.0),
                        "submit_to_decode_s": queue_s,
                        "decode_duration_s": float(decode.get("duration_s") or 0.0) if decode else 0.0,
                        "e2e_latency_s": _latency_s(request),
                    }
                )
            scenario_rows.sort(key=lambda row: row["denoise_end_wall_s"])
            first_end = scenario_rows[0]["denoise_end_wall_s"] if scenario_rows else 0.0
            previous = None
            for row in scenario_rows:
                row["denoise_end_rel_s"] = f"{row['denoise_end_wall_s'] - first_end:.6f}"
                row["previous_denoise_end_gap_s"] = (
                    "" if previous is None else f"{row['denoise_end_wall_s'] - previous:.6f}"
                )
                previous = row["denoise_end_wall_s"]
                for key in (
                    "denoise_end_wall_s",
                    "decode_start_wall_s",
                    "submit_to_decode_s",
                    "decode_duration_s",
                    "e2e_latency_s",
                ):
                    row[key] = f"{float(row[key]):.6f}"
                rows.append(row)
    return rows


def _max_in_window(values: list[float], window_s: float) -> int:
    values = sorted(values)
    best = 0
    left = 0
    for right, value in enumerate(values):
        while value - values[left] > window_s:
            left += 1
        best = max(best, right - left + 1)
    return best


def _dit_burst_rows(dit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    queues: dict[tuple[str, int], list[float]] = {}
    for row in dit_rows:
        key = (str(row["label"]), int(row["steps"]))
        grouped.setdefault(key, []).append(float(row["denoise_end_wall_s"]))
        queues.setdefault(key, []).append(float(row["submit_to_decode_s"]))
    rows: list[dict[str, Any]] = []
    for (label, steps), values in sorted(grouped.items()):
        values = sorted(values)
        gaps = [b - a for a, b in zip(values, values[1:])]
        rows.append(
            {
                "label": label,
                "steps": steps,
                "requests": len(values),
                "dit_completion_gap_mean_s": f"{_mean(gaps):.6f}",
                "dit_completion_gap_p50_s": f"{_percentile(gaps, 0.50):.6f}",
                "dit_completion_gap_p95_s": f"{_percentile(gaps, 0.95):.6f}",
                "max_completions_in_0p50s": _max_in_window(values, 0.50),
                "max_completions_in_1p00s": _max_in_window(values, 1.00),
                "decode_queue_mean_s": f"{_mean(queues[(label, steps)]):.6f}",
                "decode_queue_p95_s": f"{_percentile(queues[(label, steps)], 0.95):.6f}",
            }
        )
    return rows


def _route_rows(matrix_root: Path, labels: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in labels:
        doc = _load_result(matrix_root, label)
        for scenario in _scenarios(doc):
            steps = _steps(scenario)
            counts: dict[tuple[str, int], int] = {}
            for request in _requests(scenario):
                for event in _stage_events(request):
                    stage = str(event.get("stage"))
                    if stage not in {"encode", "denoise", "decode"}:
                        continue
                    replica_id = int(event.get("replica_id", -1))
                    counts[(stage, replica_id)] = counts.get((stage, replica_id), 0) + 1
            for (stage, replica_id), count in sorted(counts.items()):
                rows.append(
                    {
                        "label": label,
                        "steps": steps,
                        "stage": stage,
                        "replica_id": replica_id,
                        "requests": count,
                    }
                )
    return rows


def _npu_rows(matrix_root: Path, labels: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label in labels:
        for row in _read_csv(matrix_root / label / "npu_summary.csv"):
            copied = dict(row)
            copied["label"] = label
            rows.append(copied)
    return rows


def _npu_warning_rows(npu_rows: list[dict[str, str]], min_samples: int = 50) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in sorted({row["label"] for row in npu_rows}):
        label_rows = [row for row in npu_rows if row["label"] == label]
        samples = [int(float(row.get("samples") or 0)) for row in label_rows]
        if samples and min(samples) < min_samples:
            rows.append(
                {
                    "label": label,
                    "min_samples": min(samples),
                    "required_min_samples": min_samples,
                    "warning": "NPU sampling is short; use resource metrics as diagnostic only.",
                }
            )
    return rows


def _write_bar_svg(path: Path, title: str, rows: list[dict[str, Any]], value_key: str) -> None:
    width = 980
    height = 340
    margin = 54
    colors = {
        "baseline_7replica": "#666666",
        "pipeline_1decode": "#2d7dd2",
        "pipeline_1decode_dummy_decode": "#2a9d8f",
        "pipeline_1decode_stagger_0p15": "#e9c46a",
    }
    values = [float(row[value_key]) for row in rows]
    max_value = max(values) if values else 1.0
    step_x = (width - 2 * margin) / max(1, len(rows))
    bar_w = max(8.0, step_x * 0.62)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="28" font-family="sans-serif" font-size="18" font-weight="700">{title}</text>',
    ]
    baseline = height - margin
    for idx, row in enumerate(rows):
        value = float(row[value_key])
        bar_h = 0 if max_value <= 0 else value / max_value * (height - 2 * margin - 30)
        x = margin + idx * step_x
        y = baseline - bar_h
        label = str(row["label"]).replace("pipeline_", "p_")
        steps = str(row.get("steps", ""))
        color = colors.get(str(row["label"]), "#999999")
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" font-family="sans-serif" font-size="10" text-anchor="middle">{value:.3f}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{baseline + 14}" font-family="sans-serif" font-size="9" text-anchor="middle">{label}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{baseline + 26}" font-family="sans-serif" font-size="9" text-anchor="middle">{steps}step</text>')
    parts.append(f'<line x1="{margin}" x2="{width - margin}" y1="{baseline}" y2="{baseline}" stroke="#333"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def _write_manifest(path: Path, matrix_root: Path, labels: list[str]) -> None:
    lines = [
        "# Omitted Artifacts Manifest",
        "",
        "This package contains only derived CSV/SVG/Markdown artifacts.",
        "",
        "Omitted raw artifacts:",
    ]
    for label in labels:
        lines.append(f"- `{matrix_root / label}`: raw result JSON, runner/server logs, raw NPU samples, generated images.")
    path.write_text("\n".join(lines) + "\n")


def _readme(
    scenario_rows: list[dict[str, Any]],
    queue_rows: list[dict[str, Any]],
    transfer_rows: list[dict[str, Any]],
) -> str:
    indexed = {(row["label"], int(row["steps"])): row for row in scenario_rows}
    base4 = indexed.get(("baseline_7replica", 4), {})
    pipe4 = indexed.get(("pipeline_1decode", 4), {})
    dummy4 = indexed.get(("pipeline_1decode_dummy_decode", 4), {})
    stagger4 = indexed.get(("pipeline_1decode_stagger_0p15", 4), {})
    has_diagnostics = bool(dummy4 or stagger4)
    q = {
        (row["label"], int(row["steps"])): float(row["submit_to_decode_p95_s"])
        for row in queue_rows
        if int(row.get("decode_replica_id", 0)) in {-1, 0}
    }
    lines = [
        "# Qwen-Image 1decode 4-step Regression Analysis",
        "",
        "正式 pipeline 口径固定为 `1decode`：NPU0 负责 encode + 1 个 decode replica，NPU1-7 负责 7 个 DiT replica。",
    ]
    if has_diagnostics:
        lines.append("本包包含 dummy/stagger 诊断 cell；它们只用于定位原因，不作为正式候选方案。")
    else:
        lines.append("本包只比较 `baseline_7replica` 与 `pipeline_1decode`。")
    lines.extend(
        [
            "",
            "## Main 4-step Snapshot",
            "",
            "| label | throughput(img/s) | P95 latency(s) | decode queue P95(s) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in (base4, pipe4, dummy4, stagger4):
        if not row:
            continue
        label = row["label"]
        lines.append(
            f"| {label} | {float(row['throughput_img_s']):.3f} | "
            f"{float(row['latency_p95_s']):.3f} | {q.get((label, int(row['steps'])), 0.0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Trace Additions",
            "",
            "- `*_payload_export`: upstream stage output is detached/copied to CPU for cross-stage handoff.",
            "- `*_payload_import`: downstream stage receives the payload and moves tensors to its local device.",
            "- `dit_completion_events.csv`: processed per-request DiT completion wall timestamps and decode queue delay.",
            "",
            "## Transfer Rows",
            "",
        ]
    )
    if transfer_rows:
        lines.append("See `transfer_overhead_summary.csv` for payload import/export timing.")
    else:
        lines.append("No transfer trace rows were found; check whether `QWEN_IMAGE_STAGE_TRANSFER_TRACE=1` was enabled.")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `README.md`: summary and interpretation.",
            "- `run_status.csv`: run status and output locations.",
            "- `scenario_comparison.csv`: throughput and latency by scenario.",
            "- `stage_timing_breakdown.csv`: stage and transfer event timing.",
            "- `transfer_overhead_summary.csv`: payload handoff overhead.",
            "- `decode_queue_summary.csv`: submit-to-decode queue delay.",
            "- `dit_completion_burst_summary.csv`: DiT completion burst metrics.",
            "- `dit_completion_events.csv`: processed per-request DiT completion timestamps.",
            "- `route_summary.csv`: stage/replica request distribution.",
            "- `npu_summary.csv`: summarized NPU resource samples.",
            "- `npu_sampling_warnings.csv`: sampler warnings, empty when none were observed.",
            "- `throughput_comparison.svg`, `p95_comparison.svg`, `decode_queue.svg`, `stage_breakdown.svg`: safe summary figures.",
            "- `omitted_artifacts_manifest.md`: raw/log/jsonl/trace/profile omission record.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    labels = _labels(args.matrix_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_status = _manifest_rows(args.matrix_root)
    scenario_rows = _scenario_rows(args.matrix_root, labels)
    stage_rows = _stage_timing_rows(args.matrix_root, labels)
    transfer_rows = _transfer_rows(stage_rows)
    queue_rows = _decode_queue_rows(args.matrix_root, labels)
    dit_rows = _dit_completion_rows(args.matrix_root, labels)
    burst_rows = _dit_burst_rows(dit_rows)
    route_rows = _route_rows(args.matrix_root, labels)
    npu_rows = _npu_rows(args.matrix_root, labels)
    npu_warnings = _npu_warning_rows(npu_rows)

    _write_csv(args.output_dir / "run_status.csv", run_status)
    _write_csv(args.output_dir / "scenario_comparison.csv", scenario_rows)
    _write_csv(args.output_dir / "stage_timing_breakdown.csv", stage_rows)
    _write_csv(args.output_dir / "transfer_overhead_summary.csv", transfer_rows)
    _write_csv(args.output_dir / "decode_queue_summary.csv", queue_rows)
    _write_csv(args.output_dir / "dit_completion_events.csv", dit_rows)
    _write_csv(args.output_dir / "dit_completion_burst_summary.csv", burst_rows)
    _write_csv(args.output_dir / "route_summary.csv", route_rows)
    _write_csv(args.output_dir / "npu_summary.csv", npu_rows)
    _write_csv(args.output_dir / "npu_sampling_warnings.csv", npu_warnings)

    main_rows = [
        row
        for row in scenario_rows
        if row["label"] in {"baseline_7replica", "pipeline_1decode"} or int(row["steps"]) == 4
    ]
    _write_bar_svg(args.output_dir / "throughput_comparison.svg", "Throughput", main_rows, "throughput_img_s")
    _write_bar_svg(args.output_dir / "p95_comparison.svg", "P95 Latency", main_rows, "latency_p95_s")
    _write_bar_svg(args.output_dir / "decode_queue.svg", "Decode Queue P95", queue_rows, "submit_to_decode_p95_s")
    _write_bar_svg(args.output_dir / "stage_breakdown.svg", "Stage Timing Mean", stage_rows, "duration_mean_s")
    _write_manifest(args.output_dir / "omitted_artifacts_manifest.md", args.matrix_root, labels)
    (args.output_dir / "README.md").write_text(_readme(scenario_rows, queue_rows, transfer_rows))


if __name__ == "__main__":
    main()
