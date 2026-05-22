# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected in benchmark envs.
    np = None


LATENT_ASPECT_SHAPES = [
    (512, 512),
    (256, 1024),
    (1024, 256),
    (640, 640),
    (512, 800),
    (800, 512),
    (400, 1024),
    (1024, 400),
    (768, 768),
    (512, 1152),
    (1152, 512),
    (576, 1024),
    (1024, 576),
    (896, 896),
    (784, 1024),
    (1024, 784),
    (448, 1792),
    (1792, 448),
    (1024, 1024),
    (512, 2048),
    (2048, 512),
    (256, 256),
    (384, 384),
]


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
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


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values) if values else 0.0,
        "median_ms": statistics.median(values) if values else 0.0,
        "p90_ms": _percentile(values, 90),
        "p95_ms": _percentile(values, 95),
        "std_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _measurement_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    has_profile_tagged_rows = any((row.get("profile_tags") or {}).get("profile_phase") for row in rows)
    if rows and not has_profile_tagged_rows:
        return []

    out = []
    for row in rows:
        tags = row.get("profile_tags") or {}
        if tags.get("profile_phase") != "measure":
            continue
        if row.get("interrupted"):
            continue
        if row.get("denoise_ms") is None:
            continue
        out.append(row)
    return out


def _group_key(row: dict[str, Any]) -> tuple[str, str, int, float]:
    return (
        str(row.get("model")),
        str(row.get("shape_key")),
        int(row.get("batch_size") or 0),
        float(row.get("effective_batch_size") or row.get("batch_size") or 0),
    )


def build_profile_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, float], list[float]] = {}
    for row in _measurement_rows(rows):
        groups.setdefault(_group_key(row), []).append(float(row["denoise_ms"]))

    table = []
    for (model, shape_key, batch_size, effective_batch_size), values in sorted(groups.items()):
        summary = _summary(values)
        table.append(
            {
                "model": model,
                "shape_key": shape_key,
                "batch_size": batch_size,
                "effective_batch_size": effective_batch_size,
                **summary,
            }
        )
    return table


def write_profile_table_csv(table: list[dict[str, Any]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "shape_key",
        "batch_size",
        "effective_batch_size",
        "count",
        "mean_ms",
        "median_ms",
        "p90_ms",
        "p95_ms",
        "std_ms",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in table:
            writer.writerow(row)


def failed_combo_ids_from_manifest(path: str | Path | None) -> set[str]:
    return {str(row["combo_id"]) for row in manifest_failures(path) if row.get("combo_id")}


def filter_rows_by_failed_combos(rows: list[dict[str, Any]], failed_combo_ids: set[str]) -> list[dict[str, Any]]:
    if not failed_combo_ids:
        return rows
    filtered = []
    for row in rows:
        combo_id = (row.get("profile_tags") or {}).get("profile_combo_id")
        if combo_id is not None and str(combo_id) in failed_combo_ids:
            continue
        filtered.append(row)
    return filtered


def _shape_area(shape_key: str) -> float:
    try:
        width_s, height_s, frames_s = shape_key.split("x")
        return (float(width_s) * float(height_s) * float(frames_s)) / float(1024 * 1024)
    except Exception:
        return 1.0


def _parse_shape_key(shape_key: str) -> tuple[int, int, int]:
    parts = shape_key.lower().split("x")
    if len(parts) == 2:
        width_s, height_s = parts
        frames_s = "1"
    elif len(parts) == 3:
        width_s, height_s, frames_s = parts
    else:
        raise ValueError(f"invalid shape_key: {shape_key}")
    return int(width_s), int(height_s), int(frames_s)


def latent_tokens_for_shape(width: int, height: int, num_frames: int = 1) -> int:
    if width % 16 != 0 or height % 16 != 0:
        raise ValueError(f"width/height must be divisible by 16: {width}x{height}")
    return (width // 16) * (height // 16) * int(num_frames)


def aspect_ratio_for_shape(width: int, height: int) -> float:
    return max(width, height) / max(min(width, height), 1)


def _fit_fallback(table: list[dict[str, Any]]) -> dict[str, Any]:
    if np is None or len(table) < 3:
        return {
            "formula": "step_ms = c0 + c1 * area + c2 * area * batch_eff^gamma",
            "available": False,
            "reason": "numpy unavailable or too few rows",
        }

    best: dict[str, Any] | None = None
    y = np.array([float(row["median_ms"]) for row in table], dtype=float)
    for gamma in [0.5, 0.75, 1.0, 1.25, 1.5]:
        features = []
        for row in table:
            area = _shape_area(str(row["shape_key"]))
            batch_eff = max(float(row["effective_batch_size"]), 1.0)
            features.append([1.0, area, area * (batch_eff**gamma)])
        x = np.array(features, dtype=float)
        coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
        pred = x @ coeffs
        ape = np.abs((pred - y) / np.maximum(y, 1e-9)) * 100.0
        candidate = {
            "formula": "step_ms = c0 + c1 * area + c2 * area * batch_eff^gamma",
            "available": True,
            "gamma": gamma,
            "coefficients": {
                "c0": float(coeffs[0]),
                "c1": float(coeffs[1]),
                "c2": float(coeffs[2]),
            },
            "mape_pct": float(np.mean(ape)),
            "p90_ape_pct": float(np.percentile(ape, 90)),
        }
        if best is None or candidate["mape_pct"] < best["mape_pct"]:
            best = candidate
    return best or {"available": False}


def build_cost_model(table: list[dict[str, Any]]) -> dict[str, Any]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in table:
        model = str(row["model"])
        shape_key = str(row["shape_key"])
        batch_key = str(row["batch_size"])
        effective_batch_key = str(row["effective_batch_size"])
        lookup.setdefault(model, {}).setdefault(shape_key, {}).setdefault(batch_key, {})[effective_batch_key] = {
            "effective_batch_size": row["effective_batch_size"],
            "denoise_step_ms": row["median_ms"],
            "p90_ms": row["p90_ms"],
            "count": row["count"],
        }
    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "table_lookup": lookup,
        "fallback": _fit_fallback(table),
    }


def build_aspect_equivalence_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, float], list[float]] = {}
    for row in _measurement_rows(rows):
        groups.setdefault(_group_key(row), []).append(float(row["denoise_ms"]))

    table: list[dict[str, Any]] = []
    for (model, shape_key, batch_size, effective_batch_size), values in sorted(groups.items()):
        width, height, num_frames = _parse_shape_key(shape_key)
        row = {
            "model": model,
            "latent_tokens": latent_tokens_for_shape(width, height, num_frames),
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "shape_key": shape_key,
            "aspect_ratio": aspect_ratio_for_shape(width, height),
            "batch_size": batch_size,
            "effective_batch_size": effective_batch_size,
            **_summary(values),
        }
        table.append(row)

    square_baselines: dict[tuple[str, int, int, float], dict[str, Any]] = {}
    for row in table:
        if row["width"] == row["height"]:
            square_baselines[
                (
                    str(row["model"]),
                    int(row["latent_tokens"]),
                    int(row["batch_size"]),
                    float(row["effective_batch_size"]),
                )
            ] = row

    for row in table:
        baseline = square_baselines.get(
            (
                str(row["model"]),
                int(row["latent_tokens"]),
                int(row["batch_size"]),
                float(row["effective_batch_size"]),
            )
        )
        if baseline is None:
            row["delta_vs_square_pct"] = None
            row["p95_delta_vs_square_pct"] = None
            continue
        row["delta_vs_square_pct"] = (
            (float(row["median_ms"]) - float(baseline["median_ms"])) / max(float(baseline["median_ms"]), 1e-9)
        ) * 100.0
        row["p95_delta_vs_square_pct"] = (
            (float(row["p95_ms"]) - float(baseline["p95_ms"])) / max(float(baseline["p95_ms"]), 1e-9)
        ) * 100.0
    return table


def write_aspect_equivalence_csv(table: list[dict[str, Any]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "latent_tokens",
        "width",
        "height",
        "num_frames",
        "shape_key",
        "aspect_ratio",
        "batch_size",
        "effective_batch_size",
        "count",
        "mean_ms",
        "median_ms",
        "p90_ms",
        "p95_ms",
        "std_ms",
        "delta_vs_square_pct",
        "p95_delta_vs_square_pct",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in table:
            writer.writerow(row)


def _latency_model_features(row: dict[str, Any], include_aspect: bool) -> list[float]:
    latent_norm = float(row["latent_tokens"]) / 4096.0
    batch_eff = max(float(row["effective_batch_size"]), 1.0)
    features = [
        1.0,
        latent_norm,
        batch_eff,
        latent_norm * batch_eff,
    ]
    if include_aspect:
        aspect_log = abs(math.log(max(float(row["aspect_ratio"]), 1.0)))
        features.extend(
            [
                aspect_log,
                aspect_log * latent_norm,
                aspect_log * latent_norm * batch_eff,
            ]
        )
    return features


def _fit_latency_model(table: list[dict[str, Any]], *, include_aspect: bool) -> dict[str, Any]:
    if np is None or not table:
        return {"available": False, "reason": "numpy unavailable or empty table"}

    x_rows = [_latency_model_features(row, include_aspect=include_aspect) for row in table]
    if len(x_rows) <= len(x_rows[0]):
        return {"available": False, "reason": "too few rows for least-squares fit"}

    x = np.array(x_rows, dtype=float)
    y = np.array([float(row["median_ms"]) for row in table], dtype=float)
    coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coeffs
    ape = np.abs((pred - y) / np.maximum(y, 1e-9)) * 100.0
    return {
        "available": True,
        "formula": (
            "latency_ms = c0 + c1*tokens + c2*batch_eff + c3*tokens*batch_eff"
            + (
                " + c4*log(aspect) + c5*log(aspect)*tokens"
                " + c6*log(aspect)*tokens*batch_eff"
                if include_aspect
                else ""
            )
        ),
        "coefficients": [float(value) for value in coeffs.tolist()],
        "mape_pct": float(np.mean(ape)),
        "p90_ape_pct": float(np.percentile(ape, 90)),
        "max_ape_pct": float(np.max(ape)),
    }


def _fit_latency_coefficients(table: list[dict[str, Any]], *, include_aspect: bool) -> Any | None:
    if np is None or not table:
        return None
    x_rows = [_latency_model_features(row, include_aspect=include_aspect) for row in table]
    if len(x_rows) <= len(x_rows[0]):
        return None
    x = np.array(x_rows, dtype=float)
    y = np.array([float(row["median_ms"]) for row in table], dtype=float)
    coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coeffs


def _cross_validated_latency_model(table: list[dict[str, Any]], *, include_aspect: bool) -> dict[str, Any]:
    if np is None or not table:
        return {"available": False, "reason": "numpy unavailable or empty table"}

    apes = []
    groups = sorted({str(row["shape_key"]) for row in table})
    for holdout_shape in groups:
        train = [row for row in table if str(row["shape_key"]) != holdout_shape]
        test = [row for row in table if str(row["shape_key"]) == holdout_shape]
        coeffs = _fit_latency_coefficients(train, include_aspect=include_aspect)
        if coeffs is None:
            continue
        x = np.array([_latency_model_features(row, include_aspect=include_aspect) for row in test], dtype=float)
        y = np.array([float(row["median_ms"]) for row in test], dtype=float)
        pred = x @ coeffs
        apes.extend((np.abs((pred - y) / np.maximum(y, 1e-9)) * 100.0).tolist())

    if not apes:
        return {"available": False, "reason": "too few rows for leave-one-shape-out fit"}
    return {
        "available": True,
        "holdout": "shape_key",
        "mape_pct": float(np.mean(apes)),
        "p90_ape_pct": float(np.percentile(apes, 90)),
        "max_ape_pct": float(np.max(apes)),
        "num_predictions": len(apes),
    }


def _is_risk_aspect_row(row: dict[str, Any]) -> bool:
    median_delta = row.get("delta_vs_square_pct")
    p95_delta = row.get("p95_delta_vs_square_pct")
    if median_delta is None or p95_delta is None:
        return False
    return abs(float(median_delta)) > 5.0 or abs(float(p95_delta)) > 10.0


def compare_latency_models(aspect_table: list[dict[str, Any]]) -> dict[str, Any]:
    model_a = _fit_latency_model(aspect_table, include_aspect=False)
    model_b = _fit_latency_model(aspect_table, include_aspect=True)
    cv_a = _cross_validated_latency_model(aspect_table, include_aspect=False)
    cv_b = _cross_validated_latency_model(aspect_table, include_aspect=True)
    improvement = 0.0
    if model_a.get("available") and model_b.get("available"):
        improvement = float(model_a["mape_pct"]) - float(model_b["mape_pct"])
    cv_improvement = 0.0
    if cv_a.get("available") and cv_b.get("available"):
        cv_improvement = float(cv_a["mape_pct"]) - float(cv_b["mape_pct"])

    risk_shapes = [
        {
            "latent_tokens": row["latent_tokens"],
            "shape_key": row["shape_key"],
            "batch_size": row["batch_size"],
            "aspect_ratio": row["aspect_ratio"],
            "delta_vs_square_pct": row.get("delta_vs_square_pct"),
            "p95_delta_vs_square_pct": row.get("p95_delta_vs_square_pct"),
        }
        for row in aspect_table
        if _is_risk_aspect_row(row)
    ]
    aspect_needed = bool(risk_shapes) or cv_improvement >= 2.0
    return {
        "model_a_tokens_only": model_a,
        "model_b_tokens_plus_aspect": model_b,
        "mape_improvement_pct_points": improvement,
        "cross_validation": {
            "model_a_tokens_only": cv_a,
            "model_b_tokens_plus_aspect": cv_b,
            "mape_improvement_pct_points": cv_improvement,
        },
        "decision_rule": "aspect_needed when same-token deltas exceed thresholds or leave-one-shape-out MAPE improves by >= 2 pct-points",
        "aspect_needed": aspect_needed,
        "preferred_key": "shape_key" if aspect_needed else "latent_tokens",
        "fallback_features": (
            ["latent_tokens", "effective_batch_size", "aspect_ratio"]
            if aspect_needed
            else ["latent_tokens", "effective_batch_size"]
        ),
        "risk_shapes": risk_shapes,
    }


def _row_by_shape_batch(table: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["shape_key"]), int(row["batch_size"])): row for row in table}


def _shape_pair_symmetry(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_shape_batch = _row_by_shape_batch(table)
    results = []
    seen: set[tuple[str, str, int]] = set()
    for row in table:
        width, height = int(row["width"]), int(row["height"])
        if width == height:
            continue
        shape = str(row["shape_key"])
        peer_shape = f"{height}x{width}x{int(row['num_frames'])}"
        batch_size = int(row["batch_size"])
        left, right = sorted([shape, peer_shape])
        key = (left, right, batch_size)
        if key in seen:
            continue
        seen.add(key)
        peer = by_shape_batch.get((peer_shape, batch_size))
        if peer is None:
            continue
        delta = (float(row["median_ms"]) - float(peer["median_ms"])) / max(float(peer["median_ms"]), 1e-9) * 100.0
        results.append(
            {
                "shape_a": shape,
                "shape_b": peer_shape,
                "batch_size": batch_size,
                "median_delta_pct": delta,
                "abs_median_delta_pct": abs(delta),
            }
        )
    return sorted(results, key=lambda row: row["abs_median_delta_pct"], reverse=True)


def write_latent_token_report(
    *,
    output_path: str | Path,
    raw_path: str | Path,
    aspect_table: list[dict[str, Any]],
    model_comparison: dict[str, Any],
    failures: list[dict[str, Any]],
) -> None:
    by_shape_batch = _row_by_shape_batch(aspect_table)
    large_rect_checks = []
    for batch_size in [1, 2, 3, 4]:
        square = by_shape_batch.get(("1024x1024x1", batch_size))
        for shape in ["512x2048x1", "2048x512x1"]:
            row = by_shape_batch.get((shape, batch_size))
            if square is None or row is None:
                continue
            large_rect_checks.append(
                {
                    "shape_key": shape,
                    "batch_size": batch_size,
                    "delta_vs_square_pct": row.get("delta_vs_square_pct"),
                    "p95_delta_vs_square_pct": row.get("p95_delta_vs_square_pct"),
                }
            )

    symmetry = _shape_pair_symmetry(aspect_table)
    expected_symmetry_pairs = len(
        {
            tuple(sorted([f"{width}x{height}x1", f"{height}x{width}x1"]) + [str(batch_size)])
            for width, height in LATENT_ASPECT_SHAPES
            for batch_size in [1, 2, 3, 4]
            if width != height
        }
    )
    complete_symmetry_pairs = len(symmetry) >= expected_symmetry_pairs
    complete_large_rects = len(large_rect_checks) == 8
    max_symmetry_delta = symmetry[0]["abs_median_delta_pct"] if symmetry else None
    close_large_rects = complete_large_rects and all(
        abs(float(row.get("delta_vs_square_pct") or 0.0)) <= 5.0
        and abs(float(row.get("p95_delta_vs_square_pct") or 0.0)) <= 10.0
        for row in large_rect_checks
    )
    symmetric = max_symmetry_delta is not None and max_symmetry_delta <= 5.0

    lines = [
        "# Latent Tokens vs Aspect Ratio Profile Report",
        "",
        f"- raw: `{raw_path}`",
        f"- aspect rows: `{len(aspect_table)}`",
        f"- preferred_key: `{model_comparison.get('preferred_key')}`",
        f"- fallback_features: `{model_comparison.get('fallback_features')}`",
        "",
        "## 关键结论",
        "",
        (
            "- 512x2048 / 2048x512 与 1024x1024 "
            + (
                "接近。"
                if close_large_rects
                else (
                    "数据不完整，暂时不能下结论。"
                    if not complete_large_rects
                    else "不完全接近，需要保留风险标记。"
                )
            )
        ),
        (
            (
                f"- 同 latent_tokens 下，横图和竖图的最大 median 差异为 `{max_symmetry_delta:.2f}%`，"
                + (
                    "可以认为基本对称。"
                    if symmetric and complete_symmetry_pairs
                    else (
                        "已采集 pair 基本对称，但 pair 不完整，最终结论需谨慎。"
                        if symmetric
                        else "已经超过 5% 阈值。"
                    )
                )
            )
            if max_symmetry_delta is not None
            else "- 同 latent_tokens 下，缺少横竖图 pair，暂时不能判断是否对称。"
        ),
        (
            "- aspect_ratio "
            + ("需要进入 cost model。" if model_comparison.get("aspect_needed") else "暂时不需要进入 cost model。")
        ),
        (
            "- 第一版 scheduler "
            + (
                "建议继续使用 `shape_key` 作为 table lookup 主键，fallback 加入 `aspect_ratio`。"
                if model_comparison.get("aspect_needed")
                else "可以用 `latent_tokens + effective_batch_size` 替代 `shape_key` 做 fallback 估计。"
            )
        ),
        "",
        "## 1024x1024 等 token 矩形",
        "",
    ]
    if large_rect_checks:
        for row in large_rect_checks:
            lines.append(
                "- `{shape_key}` batch={batch_size}: median_delta={delta_vs_square_pct:.2f}%, "
                "p95_delta={p95_delta_vs_square_pct:.2f}%".format(**row)
            )
    else:
        lines.append("- 没有采集到 1024x1024 / 512x2048 / 2048x512 的完整对比数据。")

    lines.extend(["", "## 模型对比", ""])
    a = model_comparison.get("model_a_tokens_only") or {}
    b = model_comparison.get("model_b_tokens_plus_aspect") or {}
    if a.get("available") and b.get("available"):
        lines.append(
            "- in-sample: tokens-only MAPE={:.2f}%, tokens+aspect MAPE={:.2f}%, improvement={:.2f} pct-points.".format(
                float(a["mape_pct"]),
                float(b["mape_pct"]),
                float(model_comparison.get("mape_improvement_pct_points") or 0.0),
            )
        )
    else:
        lines.append("- 样本数不足或 numpy 不可用，未完成最小二乘拟合。")
    cv = model_comparison.get("cross_validation") or {}
    cv_a = cv.get("model_a_tokens_only") or {}
    cv_b = cv.get("model_b_tokens_plus_aspect") or {}
    if cv_a.get("available") and cv_b.get("available"):
        lines.append(
            "- leave-one-shape-out: tokens-only MAPE={:.2f}%, tokens+aspect MAPE={:.2f}%, improvement={:.2f} pct-points.".format(
                float(cv_a["mape_pct"]),
                float(cv_b["mape_pct"]),
                float(cv.get("mape_improvement_pct_points") or 0.0),
            )
        )

    lines.extend(["", "## Risk Shapes", ""])
    risk_shapes = model_comparison.get("risk_shapes") or []
    if risk_shapes:
        for row in risk_shapes[:30]:
            lines.append(
                "- `{shape_key}` batch={batch_size}, aspect={aspect_ratio:.2f}, "
                "median_delta={delta_vs_square_pct:.2f}%, p95_delta={p95_delta_vs_square_pct:.2f}%".format(
                    **row
                )
            )
        if len(risk_shapes) > 30:
            lines.append(f"- ... {len(risk_shapes) - 30} more risk rows omitted.")
    else:
        lines.append("- No risk shapes above median 5% / p95 10% thresholds.")

    lines.extend(["", "## Failures", ""])
    if failures:
        for row in failures:
            oom = "OOM" if row.get("is_oom") else "error"
            lines.append(f"- `{row.get('combo_id')}` phase={row.get('phase')} repeat={row.get('repeat')}: {oom}")
    else:
        lines.append("- No request failures found.")
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _relative_spread(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    median = statistics.median(values)
    if median <= 0:
        return 0.0
    return (max(values) - min(values)) / median


def repeat_stability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_repeat: dict[tuple[str, Any], list[float]] = {}
    for row in _measurement_rows(rows):
        tags = row.get("profile_tags") or {}
        combo = tags.get("profile_combo_id")
        repeat = tags.get("profile_repeat")
        if combo is None or repeat is None:
            continue
        per_repeat.setdefault((str(combo), repeat), []).append(float(row["denoise_ms"]))

    per_combo: dict[str, list[dict[str, float]]] = {}
    for (combo, repeat), values in per_repeat.items():
        per_combo.setdefault(combo, []).append(
            {
                "repeat": repeat,
                "median_ms": statistics.median(values),
                "p90_ms": _percentile(values, 90),
            }
        )

    stability = []
    for combo, repeat_rows in sorted(per_combo.items()):
        medians = [row["median_ms"] for row in repeat_rows]
        p90s = [row["p90_ms"] for row in repeat_rows]
        stable = _relative_spread(medians) <= 0.05 and _relative_spread(p90s) <= 0.10
        stability.append(
            {
                "profile_combo_id": combo,
                "num_repeats": len(repeat_rows),
                "median_spread_pct": _relative_spread(medians) * 100.0,
                "p90_spread_pct": _relative_spread(p90s) * 100.0,
                "stable": stable,
            }
        )
    return stability


def step_index_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values_by_step: dict[int, list[float]] = {}
    for row in _measurement_rows(rows):
        step_indices = row.get("step_indices") or []
        if len(set(step_indices)) != 1:
            continue
        values_by_step.setdefault(int(step_indices[0]), []).append(float(row["denoise_ms"]))
    if not values_by_step:
        return {"available": False}

    medians = {step: statistics.median(values) for step, values in values_by_step.items()}
    overall = statistics.median(medians.values())
    max_dev = max(abs(value - overall) / max(overall, 1e-9) for value in medians.values())
    return {
        "available": True,
        "num_steps_observed": len(medians),
        "overall_median_ms": overall,
        "max_step_median_deviation_pct": max_dev * 100.0,
        "use_step_index_in_model": max_dev > 0.05,
    }


def mixed_step_sanity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, list[float]]] = {}
    for row in rows:
        if row.get("interrupted") or row.get("denoise_ms") is None:
            continue
        tags = row.get("profile_tags") or {}
        mode = tags.get("profile_mode")
        if mode not in ("aligned", "mixed"):
            continue
        key = (str(row.get("shape_key")), int(row.get("batch_size") or 0))
        grouped.setdefault(key, {}).setdefault(str(mode), []).append(float(row["denoise_ms"]))

    results = []
    for (shape_key, batch_size), by_mode in sorted(grouped.items()):
        aligned = by_mode.get("aligned", [])
        mixed = by_mode.get("mixed", [])
        if not aligned or not mixed:
            continue
        aligned_median = statistics.median(aligned)
        mixed_median = statistics.median(mixed)
        delta_pct = (mixed_median - aligned_median) / max(aligned_median, 1e-9) * 100.0
        results.append(
            {
                "shape_key": shape_key,
                "batch_size": batch_size,
                "aligned_median_ms": aligned_median,
                "mixed_median_ms": mixed_median,
                "mixed_delta_pct": delta_pct,
                "mixed_significantly_slower": delta_pct > 10.0,
            }
        )
    return results


def manifest_failures(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None or not Path(path).exists():
        return []
    failures = []
    for row in _read_jsonl(path):
        outputs = row.get("outputs") or []
        for output in outputs:
            message = str(output.get("message") or "")
            error = output.get("error")
            if not error:
                continue
            lower_message = message.lower()
            failures.append(
                {
                    "combo_id": row.get("combo_id"),
                    "phase": row.get("phase"),
                    "repeat": row.get("repeat"),
                    "error": error,
                    "message": message,
                    "is_oom": "out of memory" in lower_message or "oom" in lower_message,
                }
            )
    return failures


def write_report(
    *,
    output_path: str | Path,
    raw_path: str | Path,
    table: list[dict[str, Any]],
    model: dict[str, Any],
    repeats: list[dict[str, Any]],
    step_stability: dict[str, Any],
    mixed_sanity: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    lines = [
        "# Diffusion Step Cost Profile Report",
        "",
        f"- raw: `{raw_path}`",
        f"- table rows: `{len(table)}`",
        f"- fallback available: `{model.get('fallback', {}).get('available')}`",
        "",
        "## Repeat Stability",
        "",
    ]
    if repeats:
        for row in repeats:
            lines.append(
                "- `{profile_combo_id}` repeats={num_repeats}, "
                "median_spread={median_spread_pct:.2f}%, "
                "p90_spread={p90_spread_pct:.2f}%, stable={stable}".format(**row)
            )
    else:
        lines.append("- No repeat tags found.")
    lines.extend(
        [
            "",
            "## Step Index Stability",
            "",
            f"- available: `{step_stability.get('available')}`",
            f"- max step median deviation: `{step_stability.get('max_step_median_deviation_pct', 0.0):.2f}%`",
            f"- use step index in model: `{step_stability.get('use_step_index_in_model', False)}`",
            "",
            "## Mixed Step Sanity",
            "",
        ]
    )
    if mixed_sanity:
        for row in mixed_sanity:
            lines.append(
                "- `{shape_key}` batch={batch_size}, aligned={aligned_median_ms:.2f} ms, "
                "mixed={mixed_median_ms:.2f} ms, delta={mixed_delta_pct:.2f}%, "
                "significantly_slower={mixed_significantly_slower}".format(**row)
            )
    else:
        lines.append("- No mixed-step sanity rows found.")
    lines.extend(
        [
            "",
            "## Failures",
            "",
        ]
    )
    if failures:
        for row in failures:
            oom = "OOM" if row.get("is_oom") else "error"
            lines.append(f"- `{row.get('combo_id')}` phase={row.get('phase')} repeat={row.get('repeat')}: {oom}")
    else:
        lines.append("- No request failures found.")
    lines.extend(
        [
            "",
            "## Scheduler Recommendation",
            "",
        ]
    )
    if step_stability.get("use_step_index_in_model"):
        lines.append("- Keep warmup/steady/final bins before wiring this profile into the scheduler.")
    else:
        lines.append("- Collapse step index and use `denoise_step_ms = f(shape, batch_size, effective_batch_size)`.")
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.input)
    table = build_profile_table(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_profile_table_csv(table, output_dir / "step_cost_profile_table.csv")
    model = build_cost_model(table)
    (output_dir / "step_cost_model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    repeats = repeat_stability(rows)
    step_stability = step_index_stability(rows)
    mixed_sanity = mixed_step_sanity(rows)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "profile_run_manifest.jsonl"
    failures = manifest_failures(manifest_path)
    (output_dir / "profile_diagnostics.json").write_text(
        json.dumps(
            {
                "repeat_stability": repeats,
                "step_index_stability": step_stability,
                "mixed_step_sanity": mixed_sanity,
                "request_failures": failures,
                "oom_combinations": sorted({row["combo_id"] for row in failures if row.get("is_oom")}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(
        output_path=output_dir / "profile_report.md",
        raw_path=args.input,
        table=table,
        model=model,
        repeats=repeats,
        step_stability=step_stability,
        mixed_sanity=mixed_sanity,
        failures=failures,
    )


def analyze_latent_aspect(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "profile_run_manifest.jsonl"
    failures = manifest_failures(manifest_path)
    failed_combo_ids = {str(row["combo_id"]) for row in failures if row.get("combo_id")}
    rows_for_model = filter_rows_by_failed_combos(rows, failed_combo_ids)

    table = build_profile_table(rows_for_model)
    write_profile_table_csv(table, output_dir / "step_cost_profile_table.csv")
    model = build_cost_model(table)
    (output_dir / "step_cost_model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")

    aspect_table = build_aspect_equivalence_table(rows_for_model)
    write_aspect_equivalence_csv(aspect_table, output_dir / "aspect_equivalence_table.csv")
    model_comparison = compare_latency_models(aspect_table)
    model_comparison["excluded_failed_combo_ids"] = sorted(failed_combo_ids)
    (output_dir / "model_comparison.json").write_text(
        json.dumps(model_comparison, indent=2),
        encoding="utf-8",
    )
    write_latent_token_report(
        output_path=output_dir / "latent_token_report.md",
        raw_path=args.input,
        aspect_table=aspect_table,
        model_comparison=model_comparison,
        failures=failures,
    )


def _send_chat_request(
    *,
    base_url: str,
    model: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    combo_id: str,
    phase: str,
    repeat: int,
    batch_size: int,
    timeout_s: int,
    profile_mode: str,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"step cost profile {combo_id} {phase} repeat {repeat}",
            }
        ],
        "extra_body": {
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "seed": seed,
            "extra_args": {
                "profile_combo_id": combo_id,
                "profile_phase": phase,
                "profile_repeat": repeat,
                "profile_batch_size": batch_size,
                "profile_shape": f"{width}x{height}",
                "profile_mode": profile_mode,
            },
        },
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start_s = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read()
    return {
        "status": resp.status,
        "latency_s": time.perf_counter() - start_s,
        "bytes": len(body),
    }


def _run_request_batch(
    *,
    base_url: str,
    model: str,
    width: int,
    height: int,
    steps: int,
    batch_size: int,
    repeat: int,
    phase: str,
    timeout_s: int,
    profile_mode: str = "aligned",
    stagger_first_s: float = 0.0,
) -> dict[str, Any]:
    combo_id = f"{width}x{height}_b{batch_size}"
    outputs = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        first_count = 1 if stagger_first_s > 0 and batch_size > 1 else 0
        for i in range(first_count):
            futures.append(
                executor.submit(
                    _send_chat_request,
                    base_url=base_url,
                    model=model,
                    width=width,
                    height=height,
                    steps=steps,
                    seed=10_000 + repeat * 100 + i,
                    combo_id=combo_id,
                    phase=phase,
                    repeat=repeat,
                    batch_size=batch_size,
                    timeout_s=timeout_s,
                    profile_mode=profile_mode,
                )
            )
        if first_count:
            time.sleep(stagger_first_s)
        for i in range(first_count, batch_size):
            futures.append(
                executor.submit(
                    _send_chat_request,
                    base_url=base_url,
                    model=model,
                    width=width,
                    height=height,
                    steps=steps,
                    seed=10_000 + repeat * 100 + i,
                    combo_id=combo_id,
                    phase=phase,
                    repeat=repeat,
                    batch_size=batch_size,
                    timeout_s=timeout_s,
                    profile_mode=profile_mode,
                )
            )
        for future in as_completed(futures):
            try:
                outputs.append(future.result())
            except Exception as exc:
                outputs.append(
                    {
                        "status": None,
                        "latency_s": 0.0,
                        "bytes": 0,
                        "error": type(exc).__name__,
                        "message": str(exc),
                    }
                )
    return {"combo_id": combo_id, "phase": phase, "repeat": repeat, "outputs": outputs}


def _has_request_errors(result: dict[str, Any]) -> bool:
    return any(output.get("error") for output in result.get("outputs", []))


def run_matrix(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "profile_run_manifest.jsonl"
    stable_initial = 0
    completed_combo_count = 0

    shapes = [tuple(map(int, shape.lower().split("x"))) for shape in args.shapes.split(",")]
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]

    with open(manifest_path, "a", encoding="utf-8") as manifest:
        for width, height in shapes:
            for batch_size in batch_sizes:
                combo_id = f"{width}x{height}_b{batch_size}"
                repeats = args.full_repeats
                if (
                    completed_combo_count >= args.initial_repeat_combos
                    and stable_initial >= args.initial_repeat_combos
                ):
                    repeats = args.reduced_repeats

                print(f"[profile] warmup {combo_id}", flush=True)
                warmup = _run_request_batch(
                    base_url=args.base_url,
                    model=args.model,
                    width=width,
                    height=height,
                    steps=args.steps,
                    batch_size=batch_size,
                    repeat=-1,
                    phase="warmup",
                    timeout_s=args.timeout_s,
                )
                manifest.write(json.dumps(warmup) + "\n")
                manifest.flush()
                if _has_request_errors(warmup):
                    print(f"[profile] skip {combo_id}: warmup failed", flush=True)
                    completed_combo_count += 1
                    continue

                for repeat in range(repeats):
                    print(f"[profile] measure {combo_id} repeat={repeat}/{repeats}", flush=True)
                    result = _run_request_batch(
                        base_url=args.base_url,
                        model=args.model,
                        width=width,
                        height=height,
                        steps=args.steps,
                        batch_size=batch_size,
                        repeat=repeat,
                        phase="measure",
                        timeout_s=args.timeout_s,
                    )
                    manifest.write(json.dumps(result) + "\n")
                    manifest.flush()
                    if _has_request_errors(result):
                        print(f"[profile] stop repeats for {combo_id}: request failed", flush=True)
                        break

                completed_combo_count += 1
                if args.raw_input_file and os.path.exists(args.raw_input_file):
                    rows = _read_jsonl(args.raw_input_file)
                    stability = [row for row in repeat_stability(rows) if row["profile_combo_id"] == combo_id]
                    if stability and stability[0]["stable"]:
                        stable_initial += 1


def _parse_shape_list(value: str | None) -> list[tuple[int, int]]:
    if not value:
        return LATENT_ASPECT_SHAPES
    shapes = []
    for item in value.split(","):
        width, height = tuple(map(int, item.lower().split("x")))
        shapes.append((width, height))
    return shapes


def _parse_batch_sizes(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _run_and_record_combo(
    *,
    manifest: Any,
    args: argparse.Namespace,
    width: int,
    height: int,
    batch_size: int,
    repeats: int,
    start_repeat: int = 0,
    run_warmup: bool = True,
) -> None:
    combo_id = f"{width}x{height}_b{batch_size}"
    if run_warmup:
        print(f"[latent-aspect] warmup {combo_id}", flush=True)
        warmup = _run_request_batch(
            base_url=args.base_url,
            model=args.model,
            width=width,
            height=height,
            steps=args.steps,
            batch_size=batch_size,
            repeat=-1,
            phase="warmup",
            timeout_s=args.timeout_s,
            profile_mode="latent_aspect",
        )
        manifest.write(json.dumps(warmup) + "\n")
        manifest.flush()
        if _has_request_errors(warmup):
            print(f"[latent-aspect] skip {combo_id}: warmup failed", flush=True)
            return

    for repeat in range(start_repeat, repeats):
        print(f"[latent-aspect] measure {combo_id} repeat={repeat}/{repeats}", flush=True)
        result = _run_request_batch(
            base_url=args.base_url,
            model=args.model,
            width=width,
            height=height,
            steps=args.steps,
            batch_size=batch_size,
            repeat=repeat,
            phase="measure",
            timeout_s=args.timeout_s,
            profile_mode="latent_aspect",
        )
        manifest.write(json.dumps(result) + "\n")
        manifest.flush()
        if _has_request_errors(result):
            print(f"[latent-aspect] stop repeats for {combo_id}: request failed", flush=True)
            break


def _unstable_square_latent_groups(raw_path: str | Path | None) -> set[int]:
    if raw_path is None or not Path(raw_path).exists():
        return set()
    rows = _read_jsonl(raw_path)
    unstable_groups: set[int] = set()
    for row in repeat_stability(rows):
        if row.get("stable") is True or int(row.get("num_repeats") or 0) < 2:
            continue
        shape_part = str(row.get("profile_combo_id") or "").split("_b", 1)[0]
        try:
            width, height = tuple(map(int, shape_part.split("x")))
        except Exception:
            continue
        if width != height:
            continue
        unstable_groups.add(latent_tokens_for_shape(width, height))
    return unstable_groups


def run_latent_aspect(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "profile_run_manifest.jsonl"
    shapes = _parse_shape_list(args.shapes)
    batch_sizes = _parse_batch_sizes(args.batch_sizes)

    with open(manifest_path, "a", encoding="utf-8") as manifest:
        for width, height in shapes:
            latent_tokens_for_shape(width, height)
            repeats = args.square_repeats if width == height else args.rect_repeats
            for batch_size in batch_sizes:
                _run_and_record_combo(
                    manifest=manifest,
                    args=args,
                    width=width,
                    height=height,
                    batch_size=batch_size,
                    repeats=repeats,
                    run_warmup=True,
                )

        if not args.supplement_unstable_rectangles:
            return

        unstable_groups = _unstable_square_latent_groups(args.raw_input_file)
        if not unstable_groups:
            print("[latent-aspect] square repeats stable; no rectangle supplement needed", flush=True)
            return

        print(
            f"[latent-aspect] supplement rectangles for unstable token groups: {sorted(unstable_groups)}",
            flush=True,
        )
        for width, height in shapes:
            if width == height:
                continue
            if latent_tokens_for_shape(width, height) not in unstable_groups:
                continue
            for batch_size in batch_sizes:
                _run_and_record_combo(
                    manifest=manifest,
                    args=args,
                    width=width,
                    height=height,
                    batch_size=batch_size,
                    repeats=args.square_repeats,
                    start_repeat=args.rect_repeats,
                    run_warmup=False,
                )


def run_mixed_sanity(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "profile_run_manifest.jsonl"
    shapes = [tuple(map(int, shape.lower().split("x"))) for shape in args.shapes.split(",")]
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]

    with open(manifest_path, "a", encoding="utf-8") as manifest:
        for width, height in shapes:
            for batch_size in batch_sizes:
                if batch_size < 2:
                    continue
                combo_id = f"{width}x{height}_b{batch_size}"
                print(f"[profile] mixed sanity {combo_id}", flush=True)
                result = _run_request_batch(
                    base_url=args.base_url,
                    model=args.model,
                    width=width,
                    height=height,
                    steps=args.steps,
                    batch_size=batch_size,
                    repeat=0,
                    phase="sanity",
                    timeout_s=args.timeout_s,
                    profile_mode="mixed",
                    stagger_first_s=args.stagger_first_s,
                )
                manifest.write(json.dumps(result) + "\n")
                manifest.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and run DiT step cost profiles.")
    sub = parser.add_subparsers(dest="command", required=True)

    aggregate_parser = sub.add_parser("aggregate", help="Aggregate step_cost_raw.jsonl into table/model/report.")
    aggregate_parser.add_argument("--input", required=True)
    aggregate_parser.add_argument("--output-dir", required=True)
    aggregate_parser.add_argument("--manifest", default=None)
    aggregate_parser.set_defaults(func=aggregate)

    latent_analyze_parser = sub.add_parser(
        "analyze-latent-aspect",
        help="Analyze whether latent_tokens can replace concrete width/height for step cost.",
    )
    latent_analyze_parser.add_argument("--input", required=True)
    latent_analyze_parser.add_argument("--output-dir", required=True)
    latent_analyze_parser.add_argument("--manifest", default=None)
    latent_analyze_parser.set_defaults(func=analyze_latent_aspect)

    run_parser = sub.add_parser("run-matrix", help="Send aligned profile traffic to a running vLLM-Omni server.")
    run_parser.add_argument("--base-url", required=True)
    run_parser.add_argument("--model", default="Qwen/Qwen-Image")
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--raw-input-file", default=None)
    run_parser.add_argument("--shapes", default="512x512,768x768,1024x1024")
    run_parser.add_argument("--batch-sizes", default="1,2,3,4")
    run_parser.add_argument("--steps", type=int, default=50)
    run_parser.add_argument("--full-repeats", type=int, default=3)
    run_parser.add_argument("--reduced-repeats", type=int, default=1)
    run_parser.add_argument("--initial-repeat-combos", type=int, default=3)
    run_parser.add_argument("--timeout-s", type=int, default=900)
    run_parser.set_defaults(func=run_matrix)

    latent_run_parser = sub.add_parser(
        "run-latent-aspect",
        help="Run the latent_tokens vs aspect-ratio profile matrix.",
    )
    latent_run_parser.add_argument("--base-url", required=True)
    latent_run_parser.add_argument("--model", default="Qwen/Qwen-Image")
    latent_run_parser.add_argument("--output-dir", required=True)
    latent_run_parser.add_argument("--raw-input-file", default=None)
    latent_run_parser.add_argument("--shapes", default=None)
    latent_run_parser.add_argument("--batch-sizes", default="1,2,3,4")
    latent_run_parser.add_argument("--steps", type=int, default=50)
    latent_run_parser.add_argument("--square-repeats", type=int, default=3)
    latent_run_parser.add_argument("--rect-repeats", type=int, default=1)
    latent_run_parser.add_argument("--timeout-s", type=int, default=900)
    latent_run_parser.add_argument(
        "--no-supplement-unstable-rectangles",
        action="store_false",
        dest="supplement_unstable_rectangles",
    )
    latent_run_parser.set_defaults(func=run_latent_aspect, supplement_unstable_rectangles=True)

    mixed_parser = sub.add_parser("run-mixed-sanity", help="Send staggered requests to create mixed-step buckets.")
    mixed_parser.add_argument("--base-url", required=True)
    mixed_parser.add_argument("--model", default="Qwen/Qwen-Image")
    mixed_parser.add_argument("--output-dir", required=True)
    mixed_parser.add_argument("--shapes", default="768x768")
    mixed_parser.add_argument("--batch-sizes", default="2,4")
    mixed_parser.add_argument("--steps", type=int, default=50)
    mixed_parser.add_argument("--stagger-first-s", type=float, default=0.5)
    mixed_parser.add_argument("--timeout-s", type=int, default=900)
    mixed_parser.set_defaults(func=run_mixed_sanity)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
