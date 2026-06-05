#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${QWEN_IMAGE_1DECODE_REPO:-/home/lzg/vllm-omni-pr4024-stage-pipeline}
VENV=${QWEN_IMAGE_1DECODE_VENV:-/home/lzg/venvs/vllm-omni-matrix-021-asc72643e}
TIMESTAMP=${QWEN_IMAGE_1DECODE_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}
MATRIX_ROOT=${QWEN_IMAGE_1DECODE_OUTDIR:-"${REPO_ROOT}/benchmark_outputs/codex-qwen-image-1decode-regression-${TIMESTAMP}"}
LATEST=${QWEN_IMAGE_1DECODE_LATEST:-"${REPO_ROOT}/benchmark_outputs/codex-qwen-image-1decode-regression.latest"}
PACKAGE_DIR=${QWEN_IMAGE_1DECODE_PACKAGE_DIR:-"${REPO_ROOT}/benchmarks/diffusion/profile_results/qwen_image_1decode_4step_regression_analysis_$(date +%Y%m%d)"}

REQUESTS=${QWEN_IMAGE_1DECODE_REQUESTS:-42}
MAIN_STEPS=${QWEN_IMAGE_1DECODE_MAIN_STEPS:-4,8,12}
HEIGHT=${QWEN_IMAGE_1DECODE_HEIGHT:-1024}
WIDTH=${QWEN_IMAGE_1DECODE_WIDTH:-1024}
TRUE_CFG_SCALE=${QWEN_IMAGE_1DECODE_TRUE_CFG_SCALE:-2.0}
RUNS=${QWEN_IMAGE_1DECODE_RUNS:-baseline_7replica,pipeline_1decode}
SUCCESS_EXIT_GRACE_S=${QWEN_IMAGE_1DECODE_SUCCESS_EXIT_GRACE_S:-30}
PIPELINE_TIMEOUT_S=${QWEN_IMAGE_1DECODE_PIPELINE_TIMEOUT_S:-7200}

mkdir -p "${MATRIX_ROOT}" "$(dirname "${LATEST}")"
ln -sfn "${MATRIX_ROOT}" "${LATEST}"

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
source "${VENV}/bin/activate"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export QUANTIZATION_CONFIG=${QUANTIZATION_CONFIG:-'"int8"'}
export PR4024_DYNAMIC_ATTENTION_BACKEND=${PR4024_DYNAMIC_ATTENTION_BACKEND:-FLASH_ATTN}
export MINDIE_SD_FA_TYPE=${MINDIE_SD_FA_TYPE:-ascend_laser_attention}
export QWEN_IMAGE_STAGE_FINE_TIMING=1
export QWEN_IMAGE_STAGE_TIMING_SYNC=1
export QWEN_IMAGE_STAGE_TRANSFER_TRACE=1
export QWEN_IMAGE_DENOISE_BATCH_TRACE=1
export QWEN_IMAGE_STAGE_PIPELINE_REQUIRE_TRACE=1
export QWEN_IMAGE_STAGE_PIPELINE_LB_POLICY=round-robin
unset VLLM_USE_MODELSCOPE

MANIFEST="${MATRIX_ROOT}/matrix_manifest.csv"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "label,kind,outdir,result_json,steps,warmup_requests,measured_requests,devices,stage_config,status,exit_code,purpose" > "${MANIFEST}"
fi

append_manifest() {
  python - "${MANIFEST}" "$@" <<'PY'
import csv
import sys

with open(sys.argv[1], "a", newline="") as handle:
    csv.writer(handle).writerow(sys.argv[2:])
PY
}

start_sampler() {
  local outdir=$1
  python benchmarks/diffusion/qwen_image_npu_sampler.py sample \
    --output "${outdir}/npu_samples_raw.csv" \
    --devices 0,1,2,3,4,5,6,7 \
    --interval-s 1.0 >"${outdir}/npu_sampler.log" 2>&1 &
  SAMPLER_PID=$!
  echo "${SAMPLER_PID}" > "${outdir}/npu_sampler.pid"
}

stop_sampler() {
  if [[ -n "${SAMPLER_PID:-}" ]]; then
    kill "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true
    SAMPLER_PID=""
  fi
}

summarize_run() {
  local label=$1
  local result_json=$2
  local outdir=$3
  stop_sampler
  python benchmarks/diffusion/qwen_image_npu_sampler.py summarize \
    --input "${outdir}/npu_samples_raw.csv" \
    --output "${outdir}/npu_summary.csv" \
    --svg "${outdir}/npu_summary.svg"
  python benchmarks/diffusion/qwen_image_pipeline_diagnostics.py \
    --label "${label}" \
    --input "${result_json}" \
    --output-dir "${outdir}/summary"
}

run_pipeline_smoke_watchdog() {
  local outdir=$1
  local result_json="${outdir}/smoke_result.json"
  python benchmarks/diffusion/run_qwen_image_stage_pipeline_smoke.py >"${outdir}/runner.log" 2>&1 &
  local smoke_pid=$!
  echo "${smoke_pid}" > "${outdir}/launcher.pid"

  local started
  started=$(date +%s)
  while kill -0 "${smoke_pid}" 2>/dev/null; do
    if python - "${result_json}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
try:
    doc = json.loads(path.read_text())
except json.JSONDecodeError:
    raise SystemExit(1)
raise SystemExit(0 if doc.get("status") == "passed" else 1)
PY
    then
      sleep "${SUCCESS_EXIT_GRACE_S}"
      if kill -0 "${smoke_pid}" 2>/dev/null; then
        kill "${smoke_pid}" 2>/dev/null || true
        wait "${smoke_pid}" 2>/dev/null || true
      else
        wait "${smoke_pid}"
      fi
      return 0
    fi
    if (( $(date +%s) - started > PIPELINE_TIMEOUT_S )); then
      kill "${smoke_pid}" 2>/dev/null || true
      wait "${smoke_pid}" 2>/dev/null || true
      return 124
    fi
    sleep 10
  done
  wait "${smoke_pid}"
}

run_baseline_7replica() {
  local label=baseline_7replica
  local outdir="${MATRIX_ROOT}/${label}"
  mkdir -p "${outdir}"
  export ASCEND_RT_VISIBLE_DEVICES=1,2,3,4,5,6,7
  export ORIGINAL_BASELINE_OUTDIR="${outdir}"
  export ORIGINAL_BASELINE_LATEST="${MATRIX_ROOT}/${label}.latest"
  export ORIGINAL_BASELINE_DEVICE_IDS=1,2,3,4,5,6,7
  export ORIGINAL_BASELINE_REQUESTS="${REQUESTS}"
  export ORIGINAL_BASELINE_STEPS_LIST="${MAIN_STEPS}"
  export ORIGINAL_BASELINE_WARMUP_REQUESTS_PER_REPLICA=1
  start_sampler "${outdir}"
  python benchmarks/diffusion/run_qwen_image_original_baseline_instrumented.py >"${outdir}/runner.log" 2>&1
  local result_json="${outdir}/original_7replica_baseline_result.json"
  summarize_run "${label}" "${result_json}" "${outdir}"
  append_manifest "${label}" baseline "${outdir}" "${result_json}" "${MAIN_STEPS}" 1 "${REQUESTS}" "1;2;3;4;5;6;7" "" passed 0 "full local encode-dit-decode baseline"
}

run_pipeline() {
  local label=$1
  local steps=$2
  local stagger=$3
  local purpose=$4
  local outdir="${MATRIX_ROOT}/${label}"
  mkdir -p "${outdir}"
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export QWEN_IMAGE_STAGE_PIPELINE_CONFIG="${REPO_ROOT}/vllm_omni/deploy/qwen_image_stage_pipeline_1x7_shared_roles.yaml"
  export QWEN_IMAGE_STAGE_PIPELINE_SMOKE_OUTDIR="${outdir}"
  export QWEN_IMAGE_STAGE_PIPELINE_HEIGHT="${HEIGHT}"
  export QWEN_IMAGE_STAGE_PIPELINE_WIDTH="${WIDTH}"
  export QWEN_IMAGE_STAGE_PIPELINE_STEPS_LIST="${steps}"
  export QWEN_IMAGE_STAGE_PIPELINE_REQUESTS="${REQUESTS}"
  export QWEN_IMAGE_STAGE_PIPELINE_WARMUP_REQUESTS=1
  export QWEN_IMAGE_STAGE_PIPELINE_TRUE_CFG_SCALE="${TRUE_CFG_SCALE}"
  export QWEN_IMAGE_STAGE_PIPELINE_STAGGER_S="${stagger}"

  start_sampler "${outdir}"
  local exit_code=0
  run_pipeline_smoke_watchdog "${outdir}" || exit_code=$?
  stop_sampler
  local result_json="${outdir}/smoke_result.json"
  if [[ "${exit_code}" == "0" ]]; then
    summarize_run "${label}" "${result_json}" "${outdir}"
    append_manifest "${label}" pipeline "${outdir}" "${result_json}" "${steps}" 1 "${REQUESTS}" "0;1;2;3;4;5;6;7" "vllm_omni/deploy/qwen_image_stage_pipeline_1x7_shared_roles.yaml" passed "${exit_code}" "${purpose}"
  else
    append_manifest "${label}" pipeline "${outdir}" "${result_json}" "${steps}" 1 "${REQUESTS}" "0;1;2;3;4;5;6;7" "vllm_omni/deploy/qwen_image_stage_pipeline_1x7_shared_roles.yaml" failed "${exit_code}" "${purpose}"
    return "${exit_code}"
  fi
}

run_one() {
  case "$1" in
    baseline_7replica)
      run_baseline_7replica
      ;;
    pipeline_1decode)
      run_pipeline pipeline_1decode "${MAIN_STEPS}" 0 "official 1decode pipeline"
      ;;
    pipeline_1decode_stagger_0p15)
      run_pipeline pipeline_1decode_stagger_0p15 4 0.15 "diagnostic: spread arrivals to reduce DiT completion bursts"
      ;;
    *)
      echo "Unknown run label: $1" >&2
      exit 2
      ;;
  esac
}

trap stop_sampler EXIT
IFS=',' read -r -a RUN_ARRAY <<< "${RUNS}"
for run_label in "${RUN_ARRAY[@]}"; do
  run_label=$(echo "${run_label}" | xargs)
  [[ -z "${run_label}" ]] && continue
  echo "=== running ${run_label} ==="
  run_one "${run_label}"
done

python benchmarks/diffusion/package_qwen_image_1decode_regression_analysis.py \
  --matrix-root "${MATRIX_ROOT}" \
  --output-dir "${PACKAGE_DIR}"

echo "${MATRIX_ROOT}"
