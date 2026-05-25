# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    return sum(values) / len(values) if values else 0.0


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


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in summary.get("rows", []) if row.get("status") == "ok"]


def _aggregate_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return list(summary.get("aggregate") or [])


def _shape_rows(
    output_dir: Path,
    *,
    workloads: list[str],
    policies: list[str],
    scales: list[float],
    repeats: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workload in workloads:
        for policy in policies:
            policy_dir = output_dir / workload / policy
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


def _distribution_rows(
    summary: dict[str, Any],
    *,
    key: str,
    value_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _summary_rows(summary):
        dist = row.get(key)
        if not isinstance(dist, dict):
            continue
        for bucket, value in sorted(dist.items(), key=lambda item: str(item[0])):
            rows.append(
                {
                    "workload": row.get("workload") or "default",
                    "policy": row.get("policy"),
                    "repeat": row.get("repeat"),
                    "slo_scale": row.get("slo_scale"),
                    value_name: bucket,
                    "count": value,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _copy_small_artifacts(
    output_dir: Path,
    bundle_dir: Path,
    *,
    workloads: list[str],
    policies: list[str],
    scales: list[float],
    repeats: list[int],
) -> None:
    for workload in workloads:
        for policy in policies:
            policy_dir = output_dir / workload / policy
            for repeat in repeats:
                for scale in scales:
                    src_dir = _scale_dir(policy_dir, repeat, scale)
                    if not src_dir.exists():
                        continue
                    dst_dir = bundle_dir / "benchmark_results" / workload / policy / f"repeat_{repeat}" / f"scale_{_scale_label(scale)}"
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
    _write_csv(
        bundle_dir / "omitted_artifacts_manifest.csv",
        rows,
        ["relative_path", "bytes", "lines", "reason"],
    )


def _metric(row: dict[str, Any], name: str) -> float:
    value = row.get(f"{name}_mean", row.get(name, 0.0))
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


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
        key=lambda row: (str(row.get("workload") or ""), float(row.get("slo_scale") or 0.0), str(row.get("policy") or "")),
    )
    groups = sorted({(str(row.get("workload") or "default"), float(row.get("slo_scale") or 0.0)) for row in rows})
    policies = sorted({str(row.get("policy") or "") for row in rows})
    if not groups or not policies:
        return

    width = max(980, 150 + len(groups) * max(70, 28 * len(policies)))
    height = 540
    left = 72
    right = 24
    top = 64
    bottom = 118
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [_metric(row, metric) for row in rows]
    max_value = max(values) if values else 0.0
    if as_percent:
        max_value = max(max_value, 0.01)
    max_value = max_value * 1.15 if max_value > 0 else 1.0
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#f59e0b"]
    by_key = {
        (str(row.get("workload") or "default"), float(row.get("slo_scale") or 0.0), str(row.get("policy") or "")): row
        for row in rows
    }
    group_w = plot_w / len(groups)
    bar_gap = 4
    bar_w = max((group_w - 16 - bar_gap * (len(policies) - 1)) / len(policies), 8)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
    ]
    for tick in range(6):
        value = max_value * tick / 5
        y = top + plot_h - plot_h * tick / 5
        label = f"{value:.0%}" if as_percent else f"{value:.2f}"
        parts.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{label}</text>')
    for group_idx, (workload, scale) in enumerate(groups):
        group_x = left + group_idx * group_w + 8
        label_x = group_x + group_w / 2 - 8
        parts.append(
            f'<text x="{label_x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="11">{workload}</text>'
        )
        parts.append(
            f'<text x="{label_x:.1f}" y="{top + plot_h + 38}" text-anchor="middle" font-family="Arial" font-size="11">scale {scale:g}</text>'
        )
        for policy_idx, policy in enumerate(policies):
            row = by_key.get((workload, scale, policy))
            value = _metric(row, metric) if row is not None else 0.0
            bar_h = plot_h * value / max_value if max_value > 0 else 0.0
            x = group_x + policy_idx * (bar_w + bar_gap)
            y = top + plot_h - bar_h
            color = colors[policy_idx % len(colors)]
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
    legend_x = left
    legend_y = height - 38
    for policy_idx, policy in enumerate(policies):
        x = legend_x + policy_idx * 210
        color = colors[policy_idx % len(colors)]
        parts.append(f'<rect x="{x}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{x + 18}" y="{legend_y}" font-family="Arial" font-size="12">{policy}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _comparison_rows(aggregate: list[dict[str, Any]], candidate_policy: str) -> list[dict[str, Any]]:
    by_key = {
        (str(row.get("workload") or "default"), float(row.get("slo_scale") or 0.0), str(row.get("policy") or "")): row
        for row in aggregate
    }
    keys = sorted({(workload, scale) for workload, scale, _ in by_key})
    rows = []
    for workload, scale in keys:
        base = by_key.get((workload, scale, "current"))
        candidate = by_key.get((workload, scale, candidate_policy))
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
            }
        )
    return rows


def _write_readme(
    bundle_dir: Path,
    *,
    workloads: list[str],
    scales: list[float],
    repeats: list[int],
    policies: list[str],
    comparison: list[dict[str, Any]],
) -> None:
    lines = [
        "# E2E SLO Workload Sweep",
        "",
        "This directory contains a lightweight, GitHub-safe result bundle. Runtime logs, raw step profiles, stagepool profiles, and trace text files are intentionally omitted and listed in `omitted_artifacts_manifest.csv`.",
        "",
        "## Experiment Matrix",
        "",
        f"- Workloads: `{', '.join(workloads)}`",
        f"- SLO scales: `{', '.join(_scale_label(scale) for scale in scales)}`",
        f"- Repeats: `{', '.join(str(repeat) for repeat in repeats)}`",
        f"- Policies: `{', '.join(policies)}`",
        "",
        "## Current vs Candidate",
        "",
        "| Workload | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            "| {workload} | {scale:g} | {miss:.2f} pp | {goodput:.2f}% | {throughput:.2f}% | {p95:.2f}% |".format(
                workload=row["workload"],
                scale=float(row["slo_scale"]),
                miss=float(row["miss_rate_delta_pp"]),
                goodput=float(row["goodput_improvement_pct"]),
                throughput=float(row["throughput_change_pct"]),
                p95=float(row["p95_latency_change_pct"]),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `ablation_summary.json`: structured summary from the raw output directory.",
            "- `ablation_table.csv`: aggregate policy x workload x scale table.",
            "- `results_by_workload_policy_scale.csv`: flattened aggregate metrics.",
            "- `current_vs_slo_no_preemption_lookup.csv`: relative deltas against `current`.",
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


def _safe_clean_bundle_dir(bundle_dir: Path) -> None:
    resolved = bundle_dir.resolve()
    if resolved.parent.name != "profile_results" or not resolved.name.startswith("e2e_slo"):
        raise ValueError(
            "--clean is only allowed for dedicated e2e_slo* bundles under "
            f"profile_results, got: {bundle_dir}"
        )
    if resolved.name in {"", ".", "..", "profile_results"}:
        raise ValueError(f"refusing to clean unsafe bundle directory: {bundle_dir}")


def _scale_labels(scales: list[float]) -> list[str]:
    return [f"{scale:g}" for scale in scales]


def _validate_summary_matrix(
    summary: dict[str, Any],
    *,
    workloads: list[str],
    policies: list[str],
    scales: list[float],
    repeats: list[int],
) -> None:
    matrix = summary.get("matrix")
    if not isinstance(matrix, dict):
        raise ValueError("ablation_summary.json is missing matrix metadata; rerun the summarizer")

    expected = {
        "workloads": workloads,
        "policies": policies,
        "scales": _scale_labels(scales),
        "repeats": [str(repeat) for repeat in repeats],
    }
    actual = {
        "workloads": [str(item) for item in matrix.get("workloads") or []],
        "policies": [str(item) for item in matrix.get("policies") or []],
        "scales": _scale_labels([float(item) for item in matrix.get("scales") or []]),
        "repeats": [str(item) for item in matrix.get("repeats") or []],
    }
    if actual != expected:
        raise ValueError(
            "summary matrix does not match package arguments: "
            f"actual={actual}, expected={expected}"
        )


def package(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    bundle_dir = Path(args.bundle_dir)
    summary_path = output_dir / "ablation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary: {summary_path}")

    if bundle_dir.exists() and args.clean:
        _safe_clean_bundle_dir(bundle_dir)
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    workloads = [item.strip() for item in args.workloads.split(",") if item.strip()]
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    scales = [float(item.strip()) for item in args.scales.split(",") if item.strip()]
    repeats = [int(item.strip()) for item in args.repeats.split(",") if item.strip()]
    summary = _read_json(summary_path)
    _validate_summary_matrix(
        summary,
        workloads=workloads,
        policies=policies,
        scales=scales,
        repeats=repeats,
    )
    aggregate = _aggregate_rows(summary)
    comparison = _comparison_rows(aggregate, args.candidate_policy)

    shutil.copy2(summary_path, bundle_dir / "ablation_summary.json")
    report = output_dir / "ablation_report.md"
    if report.exists():
        shutil.copy2(report, bundle_dir / "ablation_report.md")
    table = output_dir / "ablation_table.csv"
    if table.exists():
        shutil.copy2(table, bundle_dir / "ablation_table.csv")

    _write_csv(
        bundle_dir / "results_by_workload_policy_scale.csv",
        aggregate,
        [
            "workload",
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
            "profile_rows_before_response_filter_mean",
            "profile_rows_dropped_by_response_filter_mean",
            "p5_time_to_deadline_ms_mean",
            "mean_time_to_deadline_ms_mean",
            "stagepool_profile_rows_mean",
            "stagepool_missing_request_rows_mean",
        ],
    )
    _write_csv(
        bundle_dir / "current_vs_slo_no_preemption_lookup.csv",
        comparison,
        [
            "workload",
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
        ],
    )
    _write_csv(
        bundle_dir / "miss_by_shape.csv",
        _shape_rows(output_dir, workloads=workloads, policies=policies, scales=scales, repeats=repeats),
        [
            "workload",
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
        ["workload", "policy", "repeat", "slo_scale", "bucket_size", "count"],
    )
    _write_csv(
        bundle_dir / "stagepool_replica_distribution.csv",
        _distribution_rows(summary, key="stagepool_selected_replica_distribution", value_name="replica_id"),
        ["workload", "policy", "repeat", "slo_scale", "replica_id", "count"],
    )
    _copy_small_artifacts(
        output_dir,
        bundle_dir,
        workloads=workloads,
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
        scales=scales,
        repeats=repeats,
        policies=policies,
        comparison=comparison,
    )
    status = {
        "message": "E2E SLO result bundle packaged",
        "output_dir": str(output_dir),
        "bundle_dir": str(bundle_dir),
        "workloads": workloads,
        "policies": policies,
        "scales": scales,
        "repeats": repeats,
    }
    (bundle_dir / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Package lightweight E2E SLO sweep results for GitHub.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--workloads", default="current-mix,large-heavy,rectangular-mix,bursty")
    parser.add_argument("--policies", default="current,slo_no_preemption_lookup")
    parser.add_argument("--candidate-policy", default="slo_no_preemption_lookup")
    parser.add_argument("--scales", default="2.5,3.0,3.5,4.0")
    parser.add_argument("--repeats", default="0")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    package(args)


if __name__ == "__main__":
    main()
