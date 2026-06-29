#!/usr/bin/env python3
"""Compute pixel diffs for the Qwen-Image stage-split smoke outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def compare(lhs_path: Path, rhs_path: Path) -> dict[str, object]:
    lhs = np.asarray(Image.open(lhs_path).convert("RGB"), dtype=np.int16)
    rhs = np.asarray(Image.open(rhs_path).convert("RGB"), dtype=np.int16)
    if lhs.shape != rhs.shape:
        raise ValueError(f"Shape mismatch: {lhs_path}={lhs.shape}, {rhs_path}={rhs.shape}")

    diff = lhs - rhs
    nonzero_by_pixel = np.any(diff != 0, axis=-1)
    return {
        "shape": list(lhs.shape),
        "max_abs": int(np.max(np.abs(diff))) if diff.size else 0,
        "rmse": float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))) if diff.size else 0.0,
        "nonzero_pixels": int(np.count_nonzero(nonzero_by_pixel)),
        "nonzero_values": int(np.count_nonzero(diff)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("pairs", nargs="+", help="name=lhs.png:rhs.png")
    args = parser.parse_args()

    result = {}
    for item in args.pairs:
        name, paths = item.split("=", 1)
        lhs, rhs = paths.split(":", 1)
        result[name] = compare(Path(lhs), Path(rhs))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
