#!/usr/bin/env bash

set -e
set +u
set -o pipefail

REPO="${REPO:-/home/lzg/vllm-omni}"
VENV="${VENV:-/home/lzg/venvs/vllm-omni-matrix-021-asc72643e}"
OUTDIR="${OUTDIR:-${REPO}/benchmark_outputs/e2e_slo_first}"
MODEL="${MODEL:-Qwen/Qwen-Image}"
COST_MODEL="${COST_MODEL:-benchmarks/diffusion/profile_results/latent_token_aspect/step_cost_model.json}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
REPLICAS="${REPLICAS:-4}"
TP_SIZE="${TP_SIZE:-2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
NUM_REQUESTS="${NUM_REQUESTS:-80}"
GLOBAL_INTERARRIVAL_S="${GLOBAL_INTERARRIVAL_S:-4.25}"
SCALES="${SCALES:-2.5,4.0,5.0}"
PORT_BASE="${PORT_BASE:-18220}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-26220}"
BENCHMARK_CONCURRENCY="${BENCHMARK_CONCURRENCY:-128}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-1200}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-2400}"
ACTIVE_PID_FILE=""

mkdir -p "${OUTDIR}"

LOG="${OUTDIR}/runner.log"
STATUS_JSON="${OUTDIR}/status.json"
DEPLOY_CONFIG="${OUTDIR}/qwen_image_4replicas_tp2.yaml"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG}"
}

write_status() {
  local phase="$1"
  local message="$2"
  python - "${STATUS_JSON}" "${phase}" "${message}" "${OUTDIR}" <<'PY'
import json
import os
import sys
import time

path, phase, message, outdir = sys.argv[1:5]
payload = {
    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "phase": phase,
    "message": message,
    "outdir": outdir,
}
for rel in ["summary.json", "summary.md"]:
    payload[f"has_{rel}"] = os.path.exists(os.path.join(outdir, rel))
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

write_deploy_config() {
  python - "${DEPLOY_CONFIG}" "${DEVICES}" "${TP_SIZE}" "${MAX_NUM_SEQS}" <<'PY'
import sys

path, devices, tp_size, max_num_seqs = sys.argv[1:5]
text = f"""async_chunk: false
trust_remote_code: true

stages:
  - stage_id: 0
    stage_type: diffusion
    max_num_seqs: {int(max_num_seqs)}
    enforce_eager: false
    distributed_executor_backend: mp
    devices: "{devices}"
    parallel_config:
      pipeline_parallel_size: 1
      data_parallel_size: 1
      tensor_parallel_size: {int(tp_size)}
      enable_expert_parallel: false
      sequence_parallel_size: 1
      ulysses_degree: 1
      ring_degree: 1
      cfg_parallel_size: 1
      vae_patch_parallel_size: 1
      use_hsdp: false
      hsdp_shard_size: -1
      hsdp_replicate_size: 1
    default_sampling_params:
      num_inference_steps: 50
      height: 1024
      width: 1024
      seed: 42
"""
with open(path, "w", encoding="utf-8") as f:
    f.write(text)
PY
}

health_ok() {
  local port="$1"
  python - "${port}" <<'PY'
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
  local port="$1"
  local started
  started="$(date +%s)"
  while true; do
    if health_ok "${port}"; then
      log "service ready on port ${port}"
      return 0
    fi
    if (( $(date +%s) - started > HEALTH_TIMEOUT_S )); then
      log "service health timeout on port ${port}"
      return 1
    fi
    sleep 15
  done
}

stop_service() {
  local pid_file="$1"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]]; then
      kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
      sleep 20
      kill -9 -- "-${pid}" 2>/dev/null || kill -9 "${pid}" 2>/dev/null || true
    fi
  fi
}

cleanup_active_service() {
  if [[ -n "${ACTIVE_PID_FILE}" ]]; then
    stop_service "${ACTIVE_PID_FILE}"
    ACTIVE_PID_FILE=""
  fi
}

cleanup_on_exit() {
  local code=$?
  if [[ "${code}" -ne 0 ]]; then
    log "experiment exited with code ${code}; cleaning active service"
    cleanup_active_service
  fi
  exit "${code}"
}

trap cleanup_on_exit EXIT

make_additional_config() {
  local policy="$1"
  local raw="$2"
  python - "${policy}" "${raw}" "${COST_MODEL}" <<'PY'
import json
import sys

policy, raw, cost_model = sys.argv[1:4]
config = {
    "diffusion_step_profile": {
        "enabled": True,
        "output_path": raw,
        "sync_device": True,
        "record_rank": 0,
    }
}
if policy == "full_slo":
    config["diffusion_scheduler_policy"] = "slo"
    config["diffusion_slo_scheduler"] = {
        "step_cost_model_path": cost_model,
        "step_cost_metric": "p90_ms",
        "step_cost_formula": "qwen_image_910b_tp2_v1",
        "batch_growth_alpha": 0.6,
    }
print(json.dumps(config))
PY
}

start_service() {
  local policy="$1"
  local port="$2"
  local master_port="$3"
  local policy_dir="${OUTDIR}/${policy}"
  local raw="${policy_dir}/step_cost_raw.jsonl"
  local server_log="${policy_dir}/server.log"
  local pid_file="${policy_dir}/server.pid"
  mkdir -p "${policy_dir}"
  rm -f "${policy_dir}"/step_cost_raw*.jsonl "${server_log}" "${pid_file}"

  local additional_config
  additional_config="$(make_additional_config "${policy}" "${raw}")"
  log "starting ${policy}: port=${port}, replicas=${REPLICAS}, tp=${TP_SIZE}, devices=${DEVICES}"
  setsid bash -c "exec env \
    ASCEND_RT_VISIBLE_DEVICES='${DEVICES}' \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    python -m vllm_omni.entrypoints.cli.main serve '${MODEL}' \
      --omni \
      --host 127.0.0.1 \
      --port '${port}' \
      --tensor-parallel-size '${TP_SIZE}' \
      --max-num-seqs '${MAX_NUM_SEQS}' \
      --distributed-executor-backend mp \
      --stage-id 0 \
      --omni-master-address 127.0.0.1 \
      --omni-master-port '${master_port}' \
      --omni-dp-size-local '${REPLICAS}' \
      --step-execution \
      --deploy-config '${DEPLOY_CONFIG}' \
      --stage-init-timeout 1200 \
      --init-timeout 1800 \
      --additional-config '${additional_config}'" \
    > "${server_log}" 2>&1 < /dev/null &
  echo $! > "${pid_file}"
}

make_trace() {
  local policy="$1"
  local scale="$2"
  local dir="${OUTDIR}/${policy}/scale_${scale}"
  local trace="${dir}/trace.txt"
  mkdir -p "${dir}"
  python benchmarks/diffusion/e2e_slo_first_experiment.py make-trace \
    --output "${trace}" \
    --trace-id "${policy}_scale_${scale}" \
    --profile-policy "${policy}" \
    --num-requests "${NUM_REQUESTS}" \
    --global-interarrival-s "${GLOBAL_INTERARRIVAL_S}" \
    --slo-scale "${scale}" \
    --step-cost-model-path "${COST_MODEL}"
}

run_benchmark() {
  local policy="$1"
  local scale="$2"
  local port="$3"
  local dir="${OUTDIR}/${policy}/scale_${scale}"
  local trace="${dir}/trace.txt"
  local result="${dir}/benchmark_result.json"
  local client_log="${dir}/client.log"
  log "running ${policy} scale=${scale}"
  python benchmarks/diffusion/diffusion_benchmark_serving.py \
    --base-url "http://127.0.0.1:${port}" \
    --model "${MODEL}" \
    --endpoint /v1/chat/completions \
    --task t2i \
    --dataset trace \
    --dataset-path "${trace}" \
    --num-prompts "${NUM_REQUESTS}" \
    --num-inference-steps 50 \
    --slo-scale "${scale}" \
    --max-concurrency "${BENCHMARK_CONCURRENCY}" \
    --request-rate inf \
    --warmup-requests 0 \
    --slo \
    --output-file "${result}" \
    --disable-tqdm \
    2>&1 | tee "${client_log}"
}

run_policy() {
  local policy="$1"
  local index="$2"
  local port=$((PORT_BASE + index))
  local master_port=$((MASTER_PORT_BASE + index))
  local pid_file="${OUTDIR}/${policy}/server.pid"
  rm -rf "${OUTDIR}/${policy}"/scale_*
  write_status "starting_${policy}" "Starting ${policy}"
  start_service "${policy}" "${port}" "${master_port}"
  ACTIVE_PID_FILE="${pid_file}"
  wait_health "${port}" || { stop_service "${pid_file}"; ACTIVE_PID_FILE=""; return 1; }
  for scale in ${SCALES//,/ }; do
    make_trace "${policy}" "${scale}" || { stop_service "${pid_file}"; ACTIVE_PID_FILE=""; return 1; }
    run_benchmark "${policy}" "${scale}" "${port}" || { stop_service "${pid_file}"; ACTIVE_PID_FILE=""; return 1; }
    python benchmarks/diffusion/e2e_slo_first_experiment.py summarize --output-dir "${OUTDIR}" --scales "${SCALES}" || true
    write_status "running_${policy}" "Completed ${policy} scale=${scale}"
  done
  stop_service "${pid_file}"
  ACTIVE_PID_FILE=""
  write_status "completed_${policy}" "Completed ${policy}"
}

main() {
  setup_env
  write_deploy_config
  log "output dir: ${OUTDIR}"
  log "config: model=${MODEL}, replicas=${REPLICAS}, tp=${TP_SIZE}, global_interarrival=${GLOBAL_INTERARRIVAL_S}, scales=${SCALES}"
  run_policy current 0 || { write_status failed "current failed"; exit 1; }
  run_policy full_slo 10 || { write_status failed "full_slo failed"; exit 1; }
  python benchmarks/diffusion/e2e_slo_first_experiment.py summarize --output-dir "${OUTDIR}" --scales "${SCALES}"
  write_status completed "E2E SLO first experiment completed"
  log "completed: ${OUTDIR}"
}

main "$@"
