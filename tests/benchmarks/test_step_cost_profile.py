# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json

from benchmarks.diffusion.step_cost_profile import (
    aspect_ratio_for_shape,
    build_cost_model,
    build_aspect_equivalence_table,
    build_profile_table,
    compare_latency_models,
    latent_tokens_for_shape,
    manifest_failures,
    mixed_step_sanity,
    repeat_stability,
    step_index_stability,
    write_aspect_equivalence_csv,
    write_profile_table_csv,
)


def _row(shape: str, batch: int, repeat: int, step: int, denoise_ms: float):
    return {
        "model": "Qwen/Qwen-Image",
        "shape_key": shape,
        "batch_size": batch,
        "effective_batch_size": float(batch),
        "step_indices": [step] * batch,
        "denoise_ms": denoise_ms,
        "post_decode_ms": 0.0,
        "interrupted": False,
        "profile_tags": {
            "profile_combo_id": f"{shape}_b{batch}",
            "profile_phase": "measure",
            "profile_repeat": repeat,
        },
    }


def test_aggregate_step_cost_profile_table_and_model(tmp_path) -> None:
    rows = [
        _row("512x512x1", 1, 0, 0, 10.0),
        _row("512x512x1", 1, 0, 1, 11.0),
        _row("512x512x1", 2, 0, 0, 17.0),
        _row("512x512x1", 2, 0, 1, 19.0),
        {**_row("512x512x1", 2, 0, 2, 999.0), "profile_tags": {"profile_phase": "warmup"}},
        _row("1024x1024x1", 1, 0, 0, 40.0),
    ]

    table = build_profile_table(rows)

    one = next(row for row in table if row["shape_key"] == "512x512x1" and row["batch_size"] == 1)
    two = next(row for row in table if row["shape_key"] == "512x512x1" and row["batch_size"] == 2)
    assert one["count"] == 2
    assert one["median_ms"] == 10.5
    assert two["count"] == 2
    assert two["median_ms"] == 18.0

    csv_path = tmp_path / "table.csv"
    write_profile_table_csv(table, csv_path)
    assert "step_cost" not in csv_path.read_text()
    assert "512x512x1" in csv_path.read_text()

    model = build_cost_model(table)
    assert model["table_lookup"]["Qwen/Qwen-Image"]["512x512x1"]["1"]["1.0"]["denoise_step_ms"] == 10.5
    json.dumps(model)


def test_cost_model_keeps_distinct_effective_batch_sizes() -> None:
    table = build_profile_table(
        [
            _row("512x512x1", 1, 0, 0, 10.0),
            {**_row("512x512x1", 1, 0, 0, 20.0), "effective_batch_size": 2.0},
        ]
    )

    model = build_cost_model(table)
    lookup = model["table_lookup"]["Qwen/Qwen-Image"]["512x512x1"]["1"]

    assert lookup["1.0"]["denoise_step_ms"] == 10.0
    assert lookup["2.0"]["denoise_step_ms"] == 20.0


def test_latent_tokens_and_aspect_ratio_helpers() -> None:
    assert latent_tokens_for_shape(512, 512) == 1024
    assert latent_tokens_for_shape(1024, 1024) == 4096
    assert latent_tokens_for_shape(512, 2048) == 4096
    assert aspect_ratio_for_shape(512, 2048) == 4.0


def test_aspect_equivalence_table_and_csv(tmp_path) -> None:
    rows = [
        _row("512x512x1", 1, 0, 0, 100.0),
        _row("512x512x1", 1, 0, 1, 110.0),
        _row("256x1024x1", 1, 0, 0, 115.0),
        _row("256x1024x1", 1, 0, 1, 116.0),
    ]

    table = build_aspect_equivalence_table(rows)
    rect = next(row for row in table if row["shape_key"] == "256x1024x1")

    assert rect["latent_tokens"] == 1024
    assert rect["aspect_ratio"] == 4.0
    assert round(rect["delta_vs_square_pct"], 3) == 10.0

    csv_path = tmp_path / "aspect.csv"
    write_aspect_equivalence_csv(table, csv_path)
    assert "delta_vs_square_pct" in csv_path.read_text(encoding="utf-8")


def test_model_comparison_prefers_latent_tokens_when_rectangles_match() -> None:
    rows = [
        _row("512x512x1", 1, 0, 0, 100.0),
        _row("256x1024x1", 1, 0, 0, 101.0),
        _row("1024x256x1", 1, 0, 0, 99.0),
        _row("1024x1024x1", 2, 0, 0, 400.0),
        _row("512x2048x1", 2, 0, 0, 404.0),
        _row("2048x512x1", 2, 0, 0, 396.0),
        _row("768x768x1", 3, 0, 0, 310.0),
        _row("512x1152x1", 3, 0, 0, 312.0),
    ]

    comparison = compare_latency_models(build_aspect_equivalence_table(rows))

    assert comparison["aspect_needed"] is False
    assert comparison["preferred_key"] == "latent_tokens"


def test_model_comparison_keeps_shape_key_when_rectangles_diverge() -> None:
    rows = [
        _row("512x512x1", 1, 0, 0, 100.0),
        _row("256x1024x1", 1, 0, 0, 130.0),
        _row("1024x256x1", 1, 0, 0, 129.0),
        _row("1024x1024x1", 2, 0, 0, 400.0),
        _row("512x2048x1", 2, 0, 0, 460.0),
        _row("2048x512x1", 2, 0, 0, 459.0),
        _row("768x768x1", 3, 0, 0, 310.0),
        _row("512x1152x1", 3, 0, 0, 360.0),
    ]

    comparison = compare_latency_models(build_aspect_equivalence_table(rows))

    assert comparison["aspect_needed"] is True
    assert comparison["preferred_key"] == "shape_key"


def test_untagged_only_rows_are_not_silently_aggregated() -> None:
    row = _row("512x512x1", 1, 0, 0, 10.0)
    row["profile_tags"] = {}

    assert build_profile_table([row]) == []


def test_repeat_and_step_index_diagnostics() -> None:
    rows = [
        _row("768x768x1", 2, 0, 0, 20.0),
        _row("768x768x1", 2, 0, 1, 20.5),
        _row("768x768x1", 2, 1, 0, 20.1),
        _row("768x768x1", 2, 1, 1, 20.4),
        _row("768x768x1", 2, 2, 0, 20.2),
        _row("768x768x1", 2, 2, 1, 20.3),
    ]

    repeats = repeat_stability(rows)
    step_stability = step_index_stability(rows)

    assert repeats[0]["stable"] is True
    assert step_stability["available"] is True
    assert step_stability["use_step_index_in_model"] is False


def test_mixed_step_sanity_and_manifest_failures(tmp_path) -> None:
    aligned = _row("768x768x1", 2, 0, 0, 20.0)
    aligned["profile_tags"]["profile_mode"] = "aligned"
    mixed = _row("768x768x1", 2, 0, 1, 22.0)
    mixed["profile_tags"]["profile_mode"] = "mixed"
    mixed["profile_tags"]["profile_phase"] = "sanity"

    sanity = mixed_step_sanity([aligned, mixed])

    assert sanity[0]["mixed_delta_pct"] == 10.0
    assert sanity[0]["mixed_significantly_slower"] is False

    manifest = tmp_path / "profile_run_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "combo_id": "1024x1024_b4",
                "phase": "warmup",
                "repeat": -1,
                "outputs": [
                    {
                        "error": "RuntimeError",
                        "message": "NPU out of memory while allocating",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    failures = manifest_failures(manifest)

    assert failures[0]["combo_id"] == "1024x1024_b4"
    assert failures[0]["is_oom"] is True
