#!/usr/bin/env bash

set -e
set +u
set -o pipefail

REPO="${REPO:-/home/lzg/vllm-omni}"
VENV="${VENV:-/home/lzg/venvs/vllm-omni-matrix-021-asc72643e}"
OUTDIR="${OUTDIR:-${REPO}/benchmark_outputs/e2e_slo_ablation}"
MODEL="${MODEL:-Qwen/Qwen-Image}"
COST_MODEL="${COST_MODEL:-benchmarks/diffusion/profile_results/latent_token_aspect/step_cost_model.json}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
REPLICAS="${REPLICAS:-4}"
TP_SIZE="${TP_SIZE:-2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
NUM_REQUESTS="${NUM_REQUESTS:-80}"
GLOBAL_INTERARRIVAL_S="${GLOBAL_INTERARRIVAL_S:-4.25}"
SCALES="${SCALES:-2.5,4.0}"
REPEATS="${REPEATS:-0,1,2}"
POLICIES="${POLICIES:-current,stagepool_only,instance_only,constant_cost,formula_cost,full_slo,no_preemption,alpha0,alpha1}"
PROFILE_MODE="${PROFILE_MODE:-e2e_slo_ablation}"
PORT_BASE="${PORT_BASE:-18320}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-26320}"
BENCHMARK_CONCURRENCY="${BENCHMARK_CONCURRENCY:-128}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-2400}"
CONSTANT_STEP_MS="${CONSTANT_STEP_MS:-400}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
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
    "has_ablation_summary": os.path.exists(os.path.join(outdir, "ablation_summary.json")),
}
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

policy_index() {
  local policy="$1"
  python - "${policy}" <<'PY'
import hashlib
import sys
print(int(hashlib.md5(sys.argv[1].encode()).hexdigest()[:6], 16) % 200)
PY
}

make_additional_config() {
  local policy="$1"
  local raw="$2"
  local stagepool_raw="$3"
  python - "${policy}" "${raw}" "${stagepool_raw}" "${COST_MODEL}" "${CONSTANT_STEP_MS}" <<'PY'
import json
import sys

policy, raw, stagepool_raw, cost_model, constant_step_ms = sys.argv[1:6]
config = {
    "diffusion_step_profile": {
        "enabled": True,
        "output_path": raw,
        "sync_device": True,
        "record_rank": 0,
    },
    "diffusion_stagepool_profile": {
        "enabled": True,
        "output_path": stagepool_raw,
    },
}

lookup = {
    "step_cost_model_path": cost_model,
    "step_cost_metric": "p90_ms",
    "step_cost_formula": "qwen_image_910b_tp2_v1",
    "batch_growth_alpha": 0.6,
}

if policy == "current":
    pass
elif policy == "stagepool_only":
    config["diffusion_slo_scheduler"] = {**lookup, "enable_stagepool_slo": True}
elif policy == "instance_only":
    config["diffusion_scheduler_policy"] = "slo"
    config["diffusion_slo_scheduler"] = {**lookup, "enable_stagepool_slo": False}
elif policy == "constant_cost":
    config["diffusion_scheduler_policy"] = "slo"
    config["diffusion_slo_scheduler"] = {
        "enable_stagepool_slo": True,
        "default_step_ms": float(constant_step_ms),
        "batch_growth_alpha": 0.6,
        "ignore_request_reference_cost": True,
        "use_shape_fallback_cost": False,
    }
elif policy == "formula_cost":
    config["diffusion_scheduler_policy"] = "slo"
    config["diffusion_slo_scheduler"] = {
        "enable_stagepool_slo": True,
        "step_cost_formula": "qwen_image_910b_tp2_v1",
        "step_cost_metric": "p90_ms",
        "batch_growth_alpha": 0.6,
    }
elif policy == "full_slo":
    config["diffusion_scheduler_policy"] = "slo"
    config["diffusion_slo_scheduler"] = {**lookup, "enable_stagepool_slo": True}
elif policy == "no_preemption":
    config["diffusion_scheduler_policy"] = "slo"
    config["diffusion_slo_scheduler"] = {
        **lookup,
        "enable_stagepool_slo": True,
        "enable_step_preemption": False,
    }
elif policy == "alpha0":
    config["diffusion_scheduler_policy"] = "slo"
    config["diffusion_slo_scheduler"] = {
        "enable_stagepool_slo": True,
        "default_step_ms": float(constant_step_ms),
        "batch_growth_alpha": 0.0,
        "ignore_request_reference_cost": True,
        "use_shape_fallback_cost": True,
    }
elif policy == "alpha1":
    config["diffusion_scheduler_policy"] = "slo"
    config["diffusion_slo_scheduler"] = {
        "enable_stagepool_slo": True,
        "default_step_ms": float(constant_step_ms),
        "batch_growth_alpha": 1.0,
        "ignore_request_reference_cost": True,
        "use_shape_fallback_cost": True,
    }
else:
    raise SystemExit(f"unknown policy: {policy}")

print(json.dumps(config))
PY
}

start_service() {
  local policy="$1"
  local port="$2"
  local master_port="$3"
  local policy_dir="${OUTDIR}/${policy}"
  local raw="${policy_dir}/step_cost_raw.jsonl"
  local stagepool_raw="${policy_dir}/stagepool_profile.jsonl"
  local server_log="${policy_dir}/server.log"
  local pid_file="${policy_dir}/server.pid"
  mkdir -p "${policy_dir}"
  rm -f "${policy_dir}"/step_cost_raw*.jsonl "${stagepool_raw}" "${server_log}" "${pid_file}"
  rm -rf "${policy_dir}"/repeat_*

  local additional_config
  additional_config="$(make_additional_config "${policy}" "${raw}" "${stagepool_raw}")"
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
  local repeat="$2"
  local scale="$3"
  local dir="${OUTDIR}/${policy}/repeat_${repeat}/scale_${scale}"
  local trace="${dir}/trace.txt"
  mkdir -p "${dir}"
  python benchmarks/diffusion/e2e_slo_first_experiment.py make-trace \
    --output "${trace}" \
    --trace-id "${policy}_r${repeat}_scale_${scale}" \
    --profile-mode "${PROFILE_MODE}" \
    --profile-policy "${policy}" \
    --profile-repeat "${repeat}" \
    --num-requests "${NUM_REQUESTS}" \
    --global-interarrival-s "${GLOBAL_INTERARRIVAL_S}" \
    --slo-scale "${scale}" \
    --step-cost-model-path "${COST_MODEL}"
}

run_benchmark() {
  local policy="$1"
  local repeat="$2"
  local scale="$3"
  local port="$4"
  local dir="${OUTDIR}/${policy}/repeat_${repeat}/scale_${scale}"
  local trace="${dir}/trace.txt"
  local result="${dir}/benchmark_result.json"
  local client_log="${dir}/client.log"
  log "running ${policy} repeat=${repeat} scale=${scale}"
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

summarize_results() {
  python benchmarks/diffusion/e2e_slo_ablation_experiment.py \
    --output-dir "${OUTDIR}" \
    --policies "${POLICIES}" \
    --scales "${SCALES}" \
    --repeats "${REPEATS}" \
    --profile-mode "${PROFILE_MODE}" \
    > "${OUTDIR}/ablation_summary.stdout.json"
}

policy_results_exist() {
  local policy="$1"
  for repeat in ${REPEATS//,/ }; do
    for scale in ${SCALES//,/ }; do
      local result="${OUTDIR}/${policy}/repeat_${repeat}/scale_${scale}/benchmark_result.json"
      if [[ ! -s "${result}" ]]; then
        return 1
      fi
    done
  done
  return 0
}

run_policy() {
  local policy="$1"
  local idx
  idx="$(policy_index "${policy}")"
  local port=$((PORT_BASE + idx))
  local master_port=$((MASTER_PORT_BASE + idx))
  local pid_file="${OUTDIR}/${policy}/server.pid"

  if [[ "${SKIP_EXISTING}" == "1" ]] && policy_results_exist "${policy}"; then
    log "skipping ${policy}: existing benchmark_result.json files cover repeats=${REPEATS}, scales=${SCALES}"
    summarize_results || true
    write_status "skipped_${policy}" "Skipped ${policy}; existing results found"
    return 0
  fi

  write_status "starting_${policy}" "Starting ${policy}"
  start_service "${policy}" "${port}" "${master_port}"
  ACTIVE_PID_FILE="${pid_file}"
  wait_health "${port}" || { stop_service "${pid_file}"; ACTIVE_PID_FILE=""; return 1; }
  for repeat in ${REPEATS//,/ }; do
    for scale in ${SCALES//,/ }; do
      make_trace "${policy}" "${repeat}" "${scale}" || { stop_service "${pid_file}"; ACTIVE_PID_FILE=""; return 1; }
      run_benchmark "${policy}" "${repeat}" "${scale}" "${port}" || {
        stop_service "${pid_file}"
        ACTIVE_PID_FILE=""
        return 1
      }
      summarize_results || true
      write_status "running_${policy}" "Completed ${policy} repeat=${repeat} scale=${scale}"
    done
  done
  stop_service "${pid_file}"
  ACTIVE_PID_FILE=""
  write_status "completed_${policy}" "Completed ${policy}"
}

main() {
  setup_env
  write_deploy_config
  log "output dir: ${OUTDIR}"
  log "config: model=${MODEL}, replicas=${REPLICAS}, tp=${TP_SIZE}, requests=${NUM_REQUESTS}, interarrival=${GLOBAL_INTERARRIVAL_S}, scales=${SCALES}, repeats=${REPEATS}, policies=${POLICIES}"
  for policy in ${POLICIES//,/ }; do
    run_policy "${policy}" || { write_status failed "${policy} failed"; exit 1; }
  done
  summarize_results
  write_status completed "E2E SLO ablation completed"
  log "completed: ${OUTDIR}"
}

main "$@"
