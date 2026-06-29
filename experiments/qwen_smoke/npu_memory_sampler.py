#!/usr/bin/env python3
"""Sample NPU HBM and AICore usage into JSONL."""

from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import time
from pathlib import Path


_STOP = False


def _stop(_signum, _frame) -> None:
    global _STOP
    _STOP = True


def parse_npu_smi_info(text: str) -> dict[str, dict[str, int]]:
    devices: dict[str, dict[str, int]] = {}
    current_npu: str | None = None
    for line in text.splitlines():
        header = re.match(r"\|\s*(\d+)\s+910", line)
        if header:
            current_npu = header.group(1)
            continue
        if current_npu is None or "0000:" not in line:
            continue
        cols = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(cols) < 3:
            continue
        usage_col = cols[2]
        pairs = re.findall(r"(\d+)\s*/\s*(\d+)", usage_col)
        if not pairs:
            continue
        aicore_text = usage_col.split()[0]
        try:
            aicore = int(aicore_text)
        except ValueError:
            aicore = 0
        hbm_used, hbm_total = pairs[-1]
        devices[current_npu] = {
            "aicore_pct": aicore,
            "hbm_used_mb": int(hbm_used),
            "hbm_total_mb": int(hbm_total),
        }
    return devices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        while not _STOP:
            try:
                raw = subprocess.check_output(["npu-smi", "info"], text=True)
                sample = {
                    "ts": time.time(),
                    "devices": parse_npu_smi_info(raw),
                }
            except Exception as exc:
                sample = {"ts": time.time(), "error": repr(exc), "devices": {}}
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
