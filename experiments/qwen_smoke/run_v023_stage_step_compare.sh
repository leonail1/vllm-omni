#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
CLEAN_ROOT=${CLEAN_ROOT:-/home/lzg/vllm-omni-v023rc1-original-baseline-20260626}
CLEAN_REF=${CLEAN_REF:-v0.23.0rc1}
PR1_ROOT=${PR1_ROOT:-/home/lzg/vllm-omni-v023rc1-pr1-pipeline-20260629}
PR1_REF=${PR1_REF:-v023rc1-pipeline-runner-refactor}
MODEL=${MODEL:-/dataset/models/stage_transfer_profile_models/Qwen_Qwen-Image}
VENV=${VENV:-/home/lzg/vllm-omni-v023rc1-mixin-20260624/.venv-v023-py312}
OUT_DIR=${OUT_DIR:-${ROOT}/experiments/qwen_smoke}
TMP_DIR=${TMP_DIR:-${OUT_DIR}/tmp_stage_step_compare}
STAGE_STEP_CONFIG=${STAGE_STEP_CONFIG:-${OUT_DIR}/qwen_image_stage_split_3stage.yaml}
PROMPT=${PROMPT:-a cup of coffee on the table}
HEIGHT=${HEIGHT:-1024}
WIDTH=${WIDTH:-1024}
SEED=${SEED:-123}
WARMUP_RUNS=${WARMUP_RUNS:-1}
REPEATS=${REPEATS:-3}
NPU_SAMPLE_INTERVAL=${NPU_SAMPLE_INTERVAL:-0.5}
CANN_SET_ENV=${CANN_SET_ENV:-/usr/local/Ascend/cann-8.5.1/set_env.sh}

cd "${ROOT}"

if [[ -f "${CANN_SET_ENV}" ]]; then
  # shellcheck disable=SC1090
  set +u
  source "${CANN_SET_ENV}"
  set -u
fi

if [[ -d "${VENV}" ]]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi

if [[ ! -e "${CLEAN_ROOT}/.git" ]]; then
  git worktree add --detach "${CLEAN_ROOT}" "${CLEAN_REF}"
fi

if [[ ! -e "${PR1_ROOT}/.git" ]]; then
  git worktree add --detach "${PR1_ROOT}" "${PR1_REF}"
fi

RUNNING_SAMPLER_PID=""
cleanup() {
  if [[ -n "${RUNNING_SAMPLER_PID}" ]]; then
    kill -TERM "${RUNNING_SAMPLER_PID}" 2>/dev/null || true
    wait "${RUNNING_SAMPLER_PID}" 2>/dev/null || true
  fi
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"

run_case() {
  local python_root=$1
  local name=$2
  local steps=$3
  shift 3

  local hbm_trace="${TMP_DIR}/${name}_hbm.jsonl"
  python "${ROOT}/experiments/qwen_smoke/npu_memory_sampler.py" \
    --output "${hbm_trace}" \
    --interval "${NPU_SAMPLE_INTERVAL}" &
  local sampler_pid=$!
  RUNNING_SAMPLER_PID="${sampler_pid}"

  set +e
  (
    cd "${TMP_DIR}"
    PYTHONPATH="${python_root}:${PYTHONPATH:-}" \
      python "${ROOT}/experiments/qwen_smoke/run_omni_smoke.py" \
      --model "${MODEL}" \
      --prompt "${PROMPT}" \
      --height "${HEIGHT}" \
      --width "${WIDTH}" \
      --seed "${SEED}" \
      --steps "${steps}" \
      --warmup-runs "${WARMUP_RUNS}" \
      --repeats "${REPEATS}" \
      --output "${TMP_DIR}/${name}.png" \
      --metrics "${TMP_DIR}/${name}.json" \
      "$@"
  )
  local case_status=$?
  kill -TERM "${sampler_pid}" 2>/dev/null || true
  wait "${sampler_pid}" 2>/dev/null || true
  RUNNING_SAMPLER_PID=""
  set -e
  return "${case_status}"
}

run_case "${CLEAN_ROOT}" clean_forward_4step 4
run_case "${PR1_ROOT}" pr1_forward_4step 4
run_case "${ROOT}" pr2_forward_4step 4
run_case "${ROOT}" pr2_stage_step_4step 4 \
  --step-execution --stage-configs-path "${STAGE_STEP_CONFIG}"

run_case "${CLEAN_ROOT}" clean_forward_20step 20
run_case "${PR1_ROOT}" pr1_forward_20step 20
run_case "${ROOT}" pr2_forward_20step 20
run_case "${ROOT}" pr2_stage_step_20step 20 \
  --step-execution --stage-configs-path "${STAGE_STEP_CONFIG}"

python "${ROOT}/experiments/qwen_smoke/compare_pixel_diff.py" \
  --output "${OUT_DIR}/v023_pixel_diff.json" \
  "clean_vs_pr1_forward_4step=${TMP_DIR}/clean_forward_4step_m00.png:${TMP_DIR}/pr1_forward_4step_m00.png" \
  "clean_vs_pr2_forward_4step=${TMP_DIR}/clean_forward_4step_m00.png:${TMP_DIR}/pr2_forward_4step_m00.png" \
  "clean_vs_pr2_stage_step_4step=${TMP_DIR}/clean_forward_4step_m00.png:${TMP_DIR}/pr2_stage_step_4step_m00.png" \
  "pr1_forward_vs_pr2_forward_4step=${TMP_DIR}/pr1_forward_4step_m00.png:${TMP_DIR}/pr2_forward_4step_m00.png" \
  "pr2_forward_vs_stage_step_4step=${TMP_DIR}/pr2_forward_4step_m00.png:${TMP_DIR}/pr2_stage_step_4step_m00.png" \
  "clean_vs_pr1_forward_20step=${TMP_DIR}/clean_forward_20step_m00.png:${TMP_DIR}/pr1_forward_20step_m00.png" \
  "clean_vs_pr2_forward_20step=${TMP_DIR}/clean_forward_20step_m00.png:${TMP_DIR}/pr2_forward_20step_m00.png" \
  "clean_vs_pr2_stage_step_20step=${TMP_DIR}/clean_forward_20step_m00.png:${TMP_DIR}/pr2_stage_step_20step_m00.png" \
  "pr1_forward_vs_pr2_forward_20step=${TMP_DIR}/pr1_forward_20step_m00.png:${TMP_DIR}/pr2_forward_20step_m00.png" \
  "pr2_forward_vs_stage_step_20step=${TMP_DIR}/pr2_forward_20step_m00.png:${TMP_DIR}/pr2_stage_step_20step_m00.png"

python - "${TMP_DIR}" "${OUT_DIR}/v023_run_metrics.json" "${OUT_DIR}/v023_pixel_diff.json" "${OUT_DIR}/v023_memory_metrics.json" "${OUT_DIR}/README.md" "${CLEAN_ROOT}" "${PR1_ROOT}" "${STAGE_STEP_CONFIG}" <<'PY'
import json
import re
import subprocess
import sys
from pathlib import Path

tmp_dir = Path(sys.argv[1])
metrics_path = Path(sys.argv[2])
pixel_path = Path(sys.argv[3])
memory_path = Path(sys.argv[4])
readme_path = Path(sys.argv[5])
clean_root = Path(sys.argv[6])
pr1_root = Path(sys.argv[7])
stage_config = Path(sys.argv[8])

order = [
    "clean_forward_4step",
    "pr1_forward_4step",
    "pr2_forward_4step",
    "pr2_stage_step_4step",
    "clean_forward_20step",
    "pr1_forward_20step",
    "pr2_forward_20step",
    "pr2_stage_step_20step",
]
labels = {
    "clean_forward_4step": ("4", "clean", "forward", "false", "false"),
    "pr1_forward_4step": ("4", "pr1", "forward", "false", "false"),
    "pr2_forward_4step": ("4", "pr2", "forward", "false", "false"),
    "pr2_stage_step_4step": ("4", "pr2", "stage_step", "true", "true"),
    "clean_forward_20step": ("20", "clean", "forward", "false", "false"),
    "pr1_forward_20step": ("20", "pr1", "forward", "false", "false"),
    "pr2_forward_20step": ("20", "pr2", "forward", "false", "false"),
    "pr2_stage_step_20step": ("20", "pr2", "stage_step", "true", "true"),
}


def git_rev(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def git_rev_with_code_dirty(path: Path) -> str:
    rev = git_rev(path)
    dirty = subprocess.run(
        ["git", "-C", str(path), "diff", "--quiet", "--", "vllm_omni", "tests"],
        check=False,
    ).returncode != 0
    staged_dirty = subprocess.run(
        ["git", "-C", str(path), "diff", "--cached", "--quiet", "--", "vllm_omni", "tests"],
        check=False,
    ).returncode != 0
    return f"{rev}-dirty" if dirty or staged_dirty else rev


def load_hbm_samples(path: Path) -> list[dict]:
    if not path.exists():
        return []
    samples = []
    for line in path.read_text().splitlines():
        if line.strip():
            samples.append(json.loads(line))
    return samples


def summarize_hbm(path: Path) -> dict:
    samples = load_hbm_samples(path)
    by_device: dict[str, list[dict]] = {}
    for sample in samples:
        for device, values in sample.get("devices", {}).items():
            by_device.setdefault(device, []).append(values)
    device_summary = {}
    for device, values in by_device.items():
        used = [item["hbm_used_mb"] for item in values]
        aicore = [item.get("aicore_pct", 0) for item in values]
        total = max(item["hbm_total_mb"] for item in values)
        first = used[0]
        peak = max(used)
        last = used[-1]
        device_summary[device] = {
            "samples": len(values),
            "hbm_total_mb": total,
            "hbm_first_mb": first,
            "hbm_peak_mb": peak,
            "hbm_last_mb": last,
            "hbm_peak_delta_mb": peak - first,
            "max_aicore_pct": max(aicore) if aicore else 0,
        }
    return {
        "sample_count": len(samples),
        "trace": str(path),
        "devices": device_summary,
    }


def parse_stage_role_devices(path: Path) -> dict[str, str]:
    role_devices: dict[str, str] = {}
    block: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("- stage_id:") or stripped.startswith("stage_id:"):
            if "role" in block and "device" in block:
                role_devices[block["role"]] = block["device"]
            block = {}
        elif stripped.startswith("devices:"):
            block["device"] = stripped.split(":", 1)[1].strip().strip("'\"")
        elif stripped.startswith("stage_role:"):
            block["role"] = stripped.split(":", 1)[1].strip()
    if "role" in block and "device" in block:
        role_devices[block["role"]] = block["device"]
    return role_devices


def active_device_text(summary: dict) -> str:
    devices = summary["devices"]
    active = [
        (device, item)
        for device, item in devices.items()
        if item["hbm_peak_delta_mb"] > 256 or item["max_aicore_pct"] > 0
    ]
    if not active and devices:
        active = sorted(
            devices.items(),
            key=lambda pair: pair[1]["hbm_peak_mb"],
            reverse=True,
        )[:1]
    active.sort(key=lambda pair: pair[1]["hbm_peak_delta_mb"], reverse=True)
    return ", ".join(
        f"{device}:{item['hbm_peak_delta_mb']}MB/{item['hbm_peak_mb']}MB"
        for device, item in active
    )


def max_aicore(summary: dict) -> int:
    if not summary["devices"]:
        return 0
    return max(item["max_aicore_pct"] for item in summary["devices"].values())


data = {name: json.loads((tmp_dir / f"{name}.json").read_text()) for name in order}
memory = {name: summarize_hbm(tmp_dir / f"{name}_hbm.jsonl") for name in order}
current_root = readme_path.parents[2]
metadata = {
    "clean_root": str(clean_root),
    "clean_rev": git_rev(clean_root),
    "pr1_root": str(pr1_root),
    "pr1_rev": git_rev_with_code_dirty(pr1_root),
    "pr2_root": str(current_root),
    "pr2_rev": git_rev_with_code_dirty(current_root),
    "stage_config": str(stage_config),
}
metrics_path.write_text(
    json.dumps({"metadata": metadata, "cases": data}, ensure_ascii=False, indent=2)
    + "\n"
)

role_devices = parse_stage_role_devices(stage_config)
role_memory = {}
for name in ("pr2_stage_step_4step", "pr2_stage_step_20step"):
    role_memory[name] = {
        role: memory[name]["devices"].get(device, {})
        for role, device in role_devices.items()
    }
memory_path.write_text(
    json.dumps(
        {
            "metadata": metadata,
            "cases": memory,
            "stage_role_devices": role_devices,
            "stage_role_memory": role_memory,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)

pixel = json.loads(pixel_path.read_text())
pixel_items = (
    pixel["comparisons"]
    if isinstance(pixel, dict) and "comparisons" in pixel
    else [{"name": name, **metrics} for name, metrics in pixel.items()]
)

baseline_by_steps = {
    "4": data["clean_forward_4step"]["measured_gen_s_mean"],
    "20": data["clean_forward_20step"]["measured_gen_s_mean"],
}
metric_rows = []
for name in order:
    steps, source, label, step_execution, stage_split = labels[name]
    item = data[name]
    baseline = baseline_by_steps[steps]
    overhead_pct = ((item["measured_gen_s_mean"] / baseline) - 1.0) * 100.0
    metric_rows.append(
        "| {steps} | {source} | {label} | {step_execution} | {stage_split} | {ok} | "
        "{init:.4f} | {warmup:.4f} | {mean:.4f} | {overhead:+.2f}% | {total:.4f} |".format(
            steps=steps,
            source=source,
            label=label,
            step_execution=step_execution,
            stage_split=stage_split,
            ok=str(item["ok"]).lower(),
            init=item["init_s"],
            warmup=item["warmup_gen_s"],
            mean=item["measured_gen_s_mean"],
            overhead=overhead_pct,
            total=item["total_s"],
        )
    )

memory_rows = []
for name in order:
    steps, source, label, _, _ = labels[name]
    memory_rows.append(
        "| {steps} | {source} | {label} | {devices} | {aicore} |".format(
            steps=steps,
            source=source,
            label=label,
            devices=active_device_text(memory[name]),
            aicore=max_aicore(memory[name]),
        )
    )

role_rows = []
for name in ("pr2_stage_step_4step", "pr2_stage_step_20step"):
    steps = labels[name][0]
    for role in ("encode", "dit", "decode"):
        device = role_devices.get(role, "")
        item = memory[name]["devices"].get(device, {})
        role_rows.append(
            "| {steps} | {role} | {device} | {first} | {peak} | {delta} | {aicore} |".format(
                steps=steps,
                role=role,
                device=device,
                first=item.get("hbm_first_mb", ""),
                peak=item.get("hbm_peak_mb", ""),
                delta=item.get("hbm_peak_delta_mb", ""),
                aicore=item.get("max_aicore_pct", ""),
            )
        )

pixel_rows = []
for item in pixel_items:
    pixel_rows.append(
        "| {name} | {max_abs} | {rmse:.6f} | {nonzero_pixels} | "
        "{nonzero_values} |".format(**item)
    )

readme = f"""# Qwen-Image v0.23.0rc1 stacked PR 对比实验

模型：`/dataset/models/stage_transfer_profile_models/Qwen_Qwen-Image`

本次实验验证四组路径：

| 来源 | 模式 | step_execution | stage_split |
| --- | --- | --- | --- |
| clean | forward | false | false |
| PR1 | forward | false | false |
| PR2 | forward | false | false |
| PR2 | stage_step | true | true |

clean baseline 使用 `{metadata["clean_root"]}`，revision `{metadata["clean_rev"]}`。PR1 使用 `{metadata["pr1_root"]}`，revision `{metadata["pr1_rev"]}`。PR2 使用 `{metadata["pr2_root"]}`，revision `{metadata["pr2_rev"]}`。

每个 case 执行 `{data[order[0]]["warmup_runs"]}` 次 warmup 和 `{data[order[0]]["repeats"]}` 次 measured repeat。warmup 只用于触发首次编译/缓存，不计入 measured 统计。像素 diff 使用第 1 次 measured 输出计算；实验目录只保留 JSON 和 README，不保留图片。

## 复现入口

```bash
cd {metadata["pr2_root"]}
bash experiments/qwen_smoke/run_v023_stage_step_compare.sh
```

## 保留文件

| 文件 | 用途 |
| --- | --- |
| `README.md` | 中文实验说明和结论 |
| `v023_pixel_diff.json` | 像素差异结果 |
| `v023_run_metrics.json` | 每个 case 的耗时、代码来源和运行参数 |
| `v023_memory_metrics.json` | 每个 case 的 NPU HBM/AICore 采样汇总 |
| `run_v023_stage_step_compare.sh` | clean/PR1/PR2 对比入口 |
| `npu_memory_sampler.py` | NPU HBM/AICore 采样器 |
| `compare_pixel_diff.py` | 像素差异计算脚本 |
| `run_omni_smoke.py` | 单个 case 的 smoke runner |
| `qwen_image_stage_split_3stage.yaml` | stage_step 配置 |

## 运行耗时

| 步数 | 来源 | 模式 | step_execution | stage_split | 成功 | 初始化 s | warmup 生成 s | measured 平均 s | 相对 clean forward | 总耗时 s |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(metric_rows)}

## NPU 显存和利用率

`主动设备` 格式为 `device:峰值增量MB/峰值占用MB`，峰值增量以采样开始时的 HBM 为基线。

| 步数 | 来源 | 模式 | 主动设备 | 最大 AICore % |
| ---: | --- | --- | --- | ---: |
{chr(10).join(memory_rows)}

## PR2 stage_step 按 role 显存

| 步数 | role | device | 初始 HBM MB | 峰值 HBM MB | 峰值增量 MB | 最大 AICore % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(role_rows)}

## 像素差异

| 对比 | max_abs | rmse | nonzero_pixels | nonzero_values |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(pixel_rows)}

结论：clean、PR1 forward、PR2 forward 和 PR2 stage_step 均成功生成图片。若像素差异全为 0，则说明 pipeline/runner 重构和 stage-step 拆分均未改变输出；性能判断以 measured 平均为准，cold path 的初始化和 warmup 单独列出。
"""

readme_path.write_text(readme)
PY
