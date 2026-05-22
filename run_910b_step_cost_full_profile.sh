#!/usr/bin/env bash

set +u
set -o pipefail

REPO="${REPO:-/home/lzg/vllm-omni}"
VENV="${VENV:-/home/lzg/venvs/vllm-omni-matrix-021-asc72643e}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTDIR="${OUTDIR:-${REPO}/benchmark_outputs/step_cost_profile_full_${RUN_ID}}"
STATE_DIR="${OUTDIR}/full_experiment"
LOG="${STATE_DIR}/runner.log"
STATUS="${STATE_DIR}/status.txt"
STATUS_JSON="${STATE_DIR}/status.json"
SERVER_LOG="${STATE_DIR}/server.log"
SERVER_PID_FILE="${STATE_DIR}/server.pid"
RAW="${OUTDIR}/step_cost_raw.jsonl"

MODEL="${MODEL:-Qwen/Qwen-Image}"
PORT="${PORT:-18120}"
MASTER_PORT="${MASTER_PORT:-26120}"
DEVICES="${DEVICES:-0,1}"
TP_SIZE="${TP_SIZE:-2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
STEPS="${STEPS:-50}"
SHAPES="${SHAPES:-512x512,768x768,1024x1024}"
BATCH_SIZES="${BATCH_SIZES:-1,2,3,4}"
FULL_REPEATS="${FULL_REPEATS:-3}"
REDUCED_REPEATS="${REDUCED_REPEATS:-1}"
INITIAL_REPEAT_COMBOS="${INITIAL_REPEAT_COMBOS:-3}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-1200}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1800}"

mkdir -p "${STATE_DIR}"

write_status() {
  local phase="$1"
  local message="$2"
  printf '[%s] %s\n' "$(date '+%F %T')" "${message}" | tee -a "${LOG}"
  printf '%s\n' "${message}" > "${STATUS}"
  python - "${STATUS_JSON}" "${phase}" "${message}" "${OUTDIR}" "${RAW}" <<'PY'
import json
import os
import sys
import time

path, phase, message, outdir, raw = sys.argv[1:6]
payload = {
    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "phase": phase,
    "message": message,
    "outdir": outdir,
    "raw": raw,
}
pid_path = os.path.join(outdir, "full_experiment", "server.pid")
if os.path.exists(pid_path):
    payload["server_pid"] = open(pid_path, encoding="utf-8").read().strip()
if os.path.exists(raw):
    payload["raw_rows"] = sum(1 for line in open(raw, encoding="utf-8") if line.strip())
manifest = os.path.join(outdir, "profile_run_manifest.jsonl")
if os.path.exists(manifest):
    payload["manifest_rows"] = sum(1 for line in open(manifest, encoding="utf-8") if line.strip())
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
PY
}

setup_env() {
  cd "${REPO}" || exit 1
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /usr/local/Ascend/nnal/atb/set_env.sh
  source "${VENV}/bin/activate"
}

health_ok() {
  python - "${PORT}" <<'PY'
import sys
import urllib.request

port = sys.argv[1]
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
        raise SystemExit(0 if resp.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

wait_health() {
  local start
  start=$(date +%s)
  while true; do
    if health_ok; then
      write_status "service_ready" "服务已健康：port=${PORT}"
      return 0
    fi
    if (( $(date +%s) - start > HEALTH_TIMEOUT_S )); then
      write_status "failed" "等待服务健康超时：port=${PORT}"
      return 1
    fi
    sleep 15
  done
}

stop_service() {
  if [[ -f "${SERVER_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${SERVER_PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]]; then
      kill "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
      sleep 20
      kill -9 "-${pid}" 2>/dev/null || kill -9 "${pid}" 2>/dev/null || true
    fi
  fi
}

cleanup() {
  stop_service
}
trap cleanup EXIT

start_service() {
  write_status "starting_service" "启动 profile 服务：devices=${DEVICES}, tp=${TP_SIZE}, max_num_seqs=${MAX_NUM_SEQS}, port=${PORT}"
  rm -f "${SERVER_LOG}" "${SERVER_PID_FILE}" "${RAW}"
  local profile_config
  profile_config=$(python - "${RAW}" <<'PY'
import json
import sys

raw = sys.argv[1]
print(json.dumps({
    "diffusion_step_profile": {
        "enabled": True,
        "output_path": raw,
        "sync_device": True,
        "record_rank": 0,
    }
}))
PY
)
  setsid bash -c "exec env \
    ASCEND_RT_VISIBLE_DEVICES='${DEVICES}' \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    python -m vllm_omni.entrypoints.cli.main serve '${MODEL}' \
      --omni \
      --host 127.0.0.1 \
      --port '${PORT}' \
      --stage-id 0 \
      --omni-master-address 127.0.0.1 \
      --omni-master-port '${MASTER_PORT}' \
      --omni-dp-size-local 1 \
      --step-execution \
      --max-num-seqs '${MAX_NUM_SEQS}' \
      --tensor-parallel-size '${TP_SIZE}' \
      --stage-init-timeout 900 \
      --init-timeout 1200 \
      --additional-config '${profile_config}'" \
    > "${SERVER_LOG}" 2>&1 < /dev/null &
  echo $! > "${SERVER_PID_FILE}"
}

run_matrix() {
  write_status "running_matrix" "开始 full matrix：shapes=${SHAPES}, batch_sizes=${BATCH_SIZES}, steps=${STEPS}"
  python benchmarks/diffusion/step_cost_profile.py run-matrix \
    --base-url "http://127.0.0.1:${PORT}" \
    --model "${MODEL}" \
    --output-dir "${OUTDIR}" \
    --raw-input-file "${RAW}" \
    --shapes "${SHAPES}" \
    --batch-sizes "${BATCH_SIZES}" \
    --steps "${STEPS}" \
    --full-repeats "${FULL_REPEATS}" \
    --reduced-repeats "${REDUCED_REPEATS}" \
    --initial-repeat-combos "${INITIAL_REPEAT_COMBOS}" \
    --timeout-s "${REQUEST_TIMEOUT_S}" 2>&1 | tee -a "${LOG}"
}

run_mixed_sanity() {
  write_status "running_mixed_sanity" "开始 mixed-step sanity：768x768 batch=2,4"
  python benchmarks/diffusion/step_cost_profile.py run-mixed-sanity \
    --base-url "http://127.0.0.1:${PORT}" \
    --model "${MODEL}" \
    --output-dir "${OUTDIR}" \
    --shapes "768x768" \
    --batch-sizes "2,4" \
    --steps "${STEPS}" \
    --stagger-first-s 2.0 \
    --timeout-s "${REQUEST_TIMEOUT_S}" 2>&1 | tee -a "${LOG}"
}

aggregate_results() {
  write_status "aggregating" "聚合 profile raw：${RAW}"
  python benchmarks/diffusion/step_cost_profile.py aggregate \
    --input "${RAW}" \
    --output-dir "${OUTDIR}" \
    --manifest "${OUTDIR}/profile_run_manifest.jsonl" 2>&1 | tee -a "${LOG}"
}

main() {
  setup_env
  write_status "initializing" "full profile 输出目录：${OUTDIR}"
  start_service
  wait_health
  run_matrix
  run_mixed_sanity
  aggregate_results
  write_status "completed" "full profile 完成：${OUTDIR}"
}

main "$@"
