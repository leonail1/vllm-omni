# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = q * (len(values) - 1)
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    frac = rank - low
    return values[low] + (values[high] - values[low]) * frac


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _scenario_requests(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    requests = scenario.get("requests")
    return requests if isinstance(requests, list) else []


def _request_latency_s(request: dict[str, Any]) -> float:
    for key in ("e2e_latency_s", "latency_s", "wall_duration_s"):
        value = request.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _stage_events(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for request in requests:
        request_index = request.get("request_index")
        for event in request.get("stage_trace") or []:
            if not isinstance(event, dict):
                continue
            event = dict(event)
            event["request_index"] = request_index
            events.append(event)
    return events


def _denoise_events(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    events: list[dict[str, Any]] = []
    for request in requests:
        request_index = request.get("request_index")
        for event in request.get("denoise_batch_trace") or []:
            if not isinstance(event, dict):
                continue
            # The same batch event is attached to every request in that batch.
            # Deduplicate for per-step batch statistics, but keep request count
            # in the event itself.
            key = (
                event.get("stage_id"),
                event.get("replica_id"),
                event.get("start_s"),
                event.get("end_s"),
                event.get("batch_size"),
                event.get("total_image_tokens"),
            )
            if key in seen:
                continue
            seen.add(key)
            copied = dict(event)
            copied["request_index"] = request_index
            events.append(copied)
    return events


def _route_replica(request: dict[str, Any], stage_id: int = 1) -> int | None:
    durations = request.get("stage_durations")
    if not isinstance(durations, dict):
        return None
    value = durations.get(f"stage_{stage_id}_route_replica_id")
    if isinstance(value, (int, float)):
        return int(value)
    return None


def summarize_one(label: str, result_path: Path, output_dir: Path) -> None:
    doc = json.loads(result_path.read_text())
    scenario_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    denoise_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []

    for scenario_index, scenario in enumerate(doc.get("scenarios") or []):
        requests = _scenario_requests(scenario)
        steps = int(scenario.get("num_inference_steps") or 0)
        elapsed = float(scenario.get("measured_elapsed_s") or scenario.get("elapsed_s") or 0.0)
        latencies = [_request_latency_s(request) for request in requests]
        scenario_rows.append(
            {
                "label": label,
                "scenario_index": scenario_index,
                "steps": steps,
                "requests": len(requests),
                "elapsed_s": f"{elapsed:.6f}",
                "throughput_img_s": f"{(len(requests) / elapsed) if elapsed > 0 else 0.0:.6f}",
                "latency_mean_s": f"{_mean(latencies):.6f}",
                "latency_p50_s": f"{_percentile(latencies, 0.50):.6f}",
                "latency_p95_s": f"{_percentile(latencies, 0.95):.6f}",
            }
        )

        stage_by_name: dict[str, list[float]] = defaultdict(list)
        for event in _stage_events(requests):
            if isinstance(event.get("duration_s"), (int, float)):
                stage_by_name[str(event.get("stage"))].append(float(event["duration_s"]))
        for stage_name, durations in sorted(stage_by_name.items()):
            stage_rows.append(
                {
                    "label": label,
                    "scenario_index": scenario_index,
                    "steps": steps,
                    "stage": stage_name,
                    "events": len(durations),
                    "duration_mean_s": f"{_mean(durations):.6f}",
                    "duration_p50_s": f"{_percentile(durations, 0.50):.6f}",
                    "duration_p95_s": f"{_percentile(durations, 0.95):.6f}",
                }
            )

        denoise_by_replica: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for event in _denoise_events(requests):
            replica_id = int(event.get("replica_id", -1))
            denoise_by_replica[replica_id].append(event)
        for replica_id, events in sorted(denoise_by_replica.items()):
            durations = [float(event.get("duration_s") or 0.0) for event in events]
            batch_sizes = [float(event.get("batch_size") or 0.0) for event in events]
            token_counts = [float(event.get("total_image_tokens") or 0.0) for event in events]
            denoise_rows.append(
                {
                    "label": label,
                    "scenario_index": scenario_index,
                    "steps": steps,
                    "replica_id": replica_id,
                    "step_events": len(events),
                    "batch_mean": f"{_mean(batch_sizes):.6f}",
                    "batch_p95": f"{_percentile(batch_sizes, 0.95):.6f}",
                    "duration_mean_s": f"{_mean(durations):.6f}",
                    "duration_p95_s": f"{_percentile(durations, 0.95):.6f}",
                    "tokens_mean": f"{_mean(token_counts):.6f}",
                }
            )

        route_counts = Counter(_route_replica(request) for request in requests)
        for replica_id, count in sorted(route_counts.items(), key=lambda item: (-1 if item[0] is None else item[0])):
            route_rows.append(
                {
                    "label": label,
                    "scenario_index": scenario_index,
                    "steps": steps,
                    "stage_id": 1,
                    "replica_id": "" if replica_id is None else replica_id,
                    "requests": count,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / f"{label}_scenario_summary.csv", scenario_rows)
    _write_csv(output_dir / f"{label}_stage_summary.csv", stage_rows)
    _write_csv(output_dir / f"{label}_denoise_replica_summary.csv", denoise_rows)
    _write_csv(output_dir / f"{label}_route_summary.csv", route_rows)
    _write_throughput_svg(output_dir / f"{label}_throughput.svg", scenario_rows, label)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_throughput_svg(path: Path, rows: list[dict[str, Any]], label: str) -> None:
    width = 640
    height = 300
    margin = 48
    bar_gap = 20
    values = [float(row["throughput_img_s"]) for row in rows]
    max_value = max(values) if values else 1.0
    bar_width = (width - 2 * margin - bar_gap * max(0, len(rows) - 1)) / max(1, len(rows))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="24" y="28" font-size="16" font-family="sans-serif">{label} throughput</text>',
    ]
    for index, row in enumerate(rows):
        value = float(row["throughput_img_s"])
        bar_h = (height - 2 * margin) * value / max_value if max_value > 0 else 0.0
        x = margin + index * (bar_width + bar_gap)
        y = height - margin - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" fill="#536d8a" />')
        parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{height - 24}" text-anchor="middle">{row["steps"]} step</text>')
        parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle">{value:.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Qwen-Image pipeline/baseline diagnostic traces.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summarize_one(args.label, Path(args.input), Path(args.output_dir))


if __name__ == "__main__":
    main()
