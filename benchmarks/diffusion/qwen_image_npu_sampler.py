# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import csv
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


_STOP = False


def _handle_stop(signum: int, frame: object) -> None:
    del signum, frame
    global _STOP
    _STOP = True


def _parse_npu_smi_info(text: str) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    lines = text.splitlines()
    for index, line in enumerate(lines[:-1]):
        first = re.match(r"^\|\s*(\d+)\s+910B", line)
        if first is None:
            continue
        npu_id = int(first.group(1))
        detail = lines[index + 1]
        fields = [field.strip() for field in detail.strip().strip("|").split("|")]
        if len(fields) < 3:
            continue
        metrics = fields[2]
        numbers = re.findall(r"\d+(?:\.\d+)?", metrics)
        if len(numbers) < 4:
            continue
        rows.append(
            {
                "npu_id": npu_id,
                "aicore_pct": float(numbers[0]),
                "memory_used_mb": int(float(numbers[1])),
                "memory_total_mb": int(float(numbers[2])),
                "hbm_used_mb": int(float(numbers[3])),
                "hbm_total_mb": int(float(numbers[4])) if len(numbers) > 4 else 0,
            }
        )
    return rows


def sample(args: argparse.Namespace) -> None:
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    devices = {int(item) for item in args.devices.split(",") if item.strip()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp_s",
                "npu_id",
                "aicore_pct",
                "memory_used_mb",
                "memory_total_mb",
                "hbm_used_mb",
                "hbm_total_mb",
            ],
        )
        writer.writeheader()
        while not _STOP:
            now = time.time()
            try:
                proc = subprocess.run(
                    ["npu-smi", "info"],
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=max(5.0, args.interval_s * 4.0),
                )
                if proc.returncode == 0:
                    for row in _parse_npu_smi_info(proc.stdout):
                        if int(row["npu_id"]) not in devices:
                            continue
                        writer.writerow({"timestamp_s": f"{now:.6f}", **row})
                    handle.flush()
            except Exception:
                pass
            time.sleep(max(0.1, args.interval_s))


def summarize(args: argparse.Namespace) -> None:
    by_device: dict[int, list[dict[str, float]]] = {}
    with Path(args.input).open(newline="") as handle:
        for row in csv.DictReader(handle):
            npu_id = int(row["npu_id"])
            by_device.setdefault(npu_id, []).append(
                {
                    "aicore_pct": float(row["aicore_pct"]),
                    "hbm_used_mb": float(row["hbm_used_mb"]),
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "npu_id",
                "samples",
                "aicore_mean_pct",
                "aicore_active_mean_pct",
                "aicore_p95_pct",
                "hbm_mean_mb",
                "hbm_max_mb",
            ],
        )
        writer.writeheader()
        total_samples = 0
        for npu_id in sorted(by_device):
            rows = by_device[npu_id]
            total_samples += len(rows)
            aicore = [row["aicore_pct"] for row in rows]
            active = [value for value in aicore if value > args.active_threshold_pct]
            hbm = [row["hbm_used_mb"] for row in rows]
            writer.writerow(
                {
                    "npu_id": npu_id,
                    "samples": len(rows),
                    "aicore_mean_pct": f"{_mean(aicore):.3f}",
                    "aicore_active_mean_pct": f"{_mean(active):.3f}",
                    "aicore_p95_pct": f"{_percentile(aicore, 0.95):.3f}",
                    "hbm_mean_mb": f"{_mean(hbm):.3f}",
                    "hbm_max_mb": f"{max(hbm) if hbm else 0.0:.3f}",
                }
            )
        if total_samples == 0:
            writer.writerow(
                {
                    "npu_id": "NO_SAMPLES",
                    "samples": 0,
                    "aicore_mean_pct": "0.000",
                    "aicore_active_mean_pct": "0.000",
                    "aicore_p95_pct": "0.000",
                    "hbm_mean_mb": "0.000",
                    "hbm_max_mb": "0.000",
                }
            )

    if args.svg:
        _write_svg_summary(Path(args.svg), by_device)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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


def _write_svg_summary(path: Path, by_device: dict[int, list[dict[str, float]]]) -> None:
    width = 720
    height = 320
    margin = 48
    bar_gap = 12
    devices = sorted(by_device)
    bar_width = (width - 2 * margin - bar_gap * max(0, len(devices) - 1)) / max(1, len(devices))
    bars = []
    labels = []
    for index, npu_id in enumerate(devices):
        mean = _mean([row["aicore_pct"] for row in by_device[npu_id]])
        bar_h = (height - 2 * margin) * min(mean, 100.0) / 100.0
        x = margin + index * (bar_width + bar_gap)
        y = height - margin - bar_h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" fill="#2f6f73" />'
        )
        labels.append(f'<text x="{x + bar_width / 2:.1f}" y="{height - 24}" text-anchor="middle">NPU {npu_id}</text>')
        labels.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle">{mean:.1f}%</text>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white" />',
                '<text x="24" y="28" font-size="16" font-family="sans-serif">Mean AICore Utilization</text>',
                *bars,
                *labels,
                "</svg>",
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample and summarize Ascend NPU utilization for Qwen-Image experiments.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--output", required=True)
    sample_parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    sample_parser.add_argument("--interval-s", type=float, default=1.0)
    sample_parser.set_defaults(func=sample)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--input", required=True)
    summarize_parser.add_argument("--output", required=True)
    summarize_parser.add_argument("--svg")
    summarize_parser.add_argument("--active-threshold-pct", type=float, default=1.0)
    summarize_parser.set_defaults(func=summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
