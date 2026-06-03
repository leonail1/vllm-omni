#!/usr/bin/env bash

set -e
set +u
set -o pipefail

REPO="${REPO:-/home/lzg/vllm-omni}"
VENV="${VENV:-/home/lzg/venvs/vllm-omni-matrix-021-asc72643e}"
OUTDIR="${OUTDIR:-${REPO}/benchmark_outputs/e2e_slo_frontier}"
MODEL="${MODEL:-Qwen/Qwen-Image}"
COST_MODEL="${COST_MODEL:-benchmarks/diffusion/profile_results/latent_token_aspect/step_cost_model.json}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
REPLICAS="${REPLICAS:-4}"
TP_SIZE="${TP_SIZE:-2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
QUANTIZATION_CONFIG="${QUANTIZATION_CONFIG:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
STATIC_ATTENTION_BACKEND="${STATIC_ATTENTION_BACKEND:-TORCH_SDPA}"
NUM_REQUESTS="${NUM_REQUESTS:-60}"
INTERARRIVALS="${INTERARRIVALS:-4.25,2.5}"
SCALES="${SCALES:-4.0,6.0}"
REPEATS="${REPEATS:-0}"
WORKLOADS="${WORKLOADS:-current-mix,shape-grouped-current-mix}"
POLICIES="${POLICIES:-current,slo_no_preemption_lookup}"
CANDIDATE_POLICY="${CANDIDATE_POLICY:-}"
if [[ -z "${CANDIDATE_POLICY}" ]]; then
  CANDIDATE_POLICY="slo_no_preemption_lookup"
  if [[ ",${POLICIES}," == *",slo_no_preemption_guarded,"* ]]; then
    CANDIDATE_POLICY="slo_no_preemption_guarded"
  fi
  if [[ ",${POLICIES}," == *",slo_no_preemption_token_guarded,"* ]]; then
    CANDIDATE_POLICY="slo_no_preemption_token_guarded"
  fi
  if [[ ",${POLICIES}," == *",slo_no_preemption_token_adaptive,"* ]]; then
    CANDIDATE_POLICY="slo_no_preemption_token_adaptive"
  fi
  if [[ ",${POLICIES}," == *",slo_no_preemption_token_objective,"* ]]; then
    CANDIDATE_POLICY="slo_no_preemption_token_objective"
  fi
  if [[ ",${POLICIES}," == *",slo_token_stagepool_objective,"* ]]; then
    CANDIDATE_POLICY="slo_token_stagepool_objective"
  fi
  if [[ ",${POLICIES}," == *",slo_token_step_preemptive,"* ]]; then
    CANDIDATE_POLICY="slo_token_step_preemptive"
  fi
  if [[ ",${POLICIES}," == *",slo_step_preemptive_token,"* ]]; then
    CANDIDATE_POLICY="slo_step_preemptive_token"
  fi
  if [[ ",${POLICIES}," == *",slo_token_preemptive,"* ]]; then
    CANDIDATE_POLICY="slo_token_preemptive"
  fi
fi
PROFILE_MODE="${PROFILE_MODE:-e2e_slo_frontier}"
PORT_BASE="${PORT_BASE:-18420}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-26420}"
BENCHMARK_CONCURRENCY="${BENCHMARK_CONCURRENCY:-128}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-2400}"
CONSTANT_STEP_MS="${CONSTANT_STEP_MS:-400}"
STAGEPOOL_LAXITY_WINDOW_MS="${STAGEPOOL_LAXITY_WINDOW_MS:-1500}"
STAGEPOOL_PACK_MIN_LAXITY_MS="${STAGEPOOL_PACK_MIN_LAXITY_MS:-1000}"
NO_PREEMPTION_ADMISSION_GUARD="${NO_PREEMPTION_ADMISSION_GUARD:-1}"
GUARDED_STAGEPOOL_LAXITY_WINDOW_MS="${GUARDED_STAGEPOOL_LAXITY_WINDOW_MS:-500}"
GUARDED_STAGEPOOL_PACK_MAX_QUEUE_LENGTH="${GUARDED_STAGEPOOL_PACK_MAX_QUEUE_LENGTH:-2}"
GUARDED_STAGEPOOL_PACK_MAX_MATCHING_BUCKET_SIZE="${GUARDED_STAGEPOOL_PACK_MAX_MATCHING_BUCKET_SIZE:-2}"
GUARDED_NO_PREEMPTION_ADMISSION_MAX_BATCH_SIZE="${GUARDED_NO_PREEMPTION_ADMISSION_MAX_BATCH_SIZE:-3}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
ACTIVE_PID_FILE=""

mkdir -p "${OUTDIR}"

LOG="${OUTDIR}/runner.log"
STATUS_JSON="${OUTDIR}/status.json"
DEPLOY_CONFIG="${OUTDIR}/qwen_image_${REPLICAS}replicas_tp${TP_SIZE}.yaml"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG}"
}

validate_safe_token() {
  local kind="$1"
  local value="$2"
  if [[ -z "${value}" || "${value}" == *"/"* || "${value}" == *".."* ]]; then
    log "invalid ${kind}: ${value}"
    exit 2
  fi
}

validate_workload() {
  local workload="$1"
  validate_safe_token "workload" "${workload}"
  case "${workload}" in
    current-mix|shape-grouped-current-mix|large-heavy|rectangular-mix|bursty) ;;
    *)
      log "unknown workload: ${workload}"
      exit 2
      ;;
  esac
}

validate_policy() {
  local policy="$1"
  validate_safe_token "policy" "${policy}"
  case "${policy}" in
    current|stagepool_only|instance_only|constant_cost|formula_cost|full_slo|no_preemption|slo_no_preemption_lookup|slo_no_preemption_guarded|slo_no_preemption_token_guarded|slo_no_preemption_token_adaptive|slo_no_preemption_token_objective|slo_token_stagepool_objective|slo_token_step_preemptive|slo_step_preemptive_token|slo_token_preemptive|pr4024_dynamic|alpha0|alpha1) ;;
    *)
      log "unknown policy: ${policy}"
      exit 2
      ;;
  esac
}

validate_repeat() {
  local repeat="$1"
  validate_safe_token "repeat" "${repeat}"
  if [[ ! "${repeat}" =~ ^[0-9]+$ ]]; then
    log "invalid repeat: ${repeat}"
    exit 2
  fi
}

validate_decimal() {
  local kind="$1"
  local value="$2"
  validate_safe_token "${kind}" "${value}"
  if [[ ! "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    log "invalid ${kind}: ${value}"
    exit 2
  fi
}

decimal_label() {
  local value="$1"
  python - "${value}" <<'PY'
import sys
print(f"{float(sys.argv[1]):g}")
PY
}

interarrival_token() {
  local value="$1"
  decimal_label "${value}" | sed 's/\./p/g'
}

load_label() {
  local interarrival="$1"
  printf 'ia_%s' "$(interarrival_token "${interarrival}")"
}

validate_matrix_args() {
  local workload
  for workload in ${WORKLOADS//,/ }; do
    validate_workload "${workload}"
  done
  local policy
  for policy in ${POLICIES//,/ }; do
    validate_policy "${policy}"
  done
  validate_policy "${CANDIDATE_POLICY}"
  local repeat
  for repeat in ${REPEATS//,/ }; do
    validate_repeat "${repeat}"
  done
  local scale
  for scale in ${SCALES//,/ }; do
    validate_decimal "scale" "${scale}"
  done
  local interarrival
  for interarrival in ${INTERARRIVALS//,/ }; do
    validate_decimal "interarrival" "${interarrival}"
  done
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
    "has_frontier_summary": os.path.exists(os.path.join(outdir, "frontier_summary.json")),
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
  local venv_pythonpath=""
  venv_pythonpath="$(python - <<'PY'
import sysconfig

paths = []
for key in ("purelib", "platlib"):
    path = sysconfig.get_paths().get(key)
    if path and path not in paths:
        paths.append(path)
print(":".join(paths))
PY
)"
  local runtime_pythonpath=""
  local entry
  IFS=':' read -r -a pythonpath_entries <<< "${PYTHONPATH:-}"
  for entry in "${pythonpath_entries[@]}"; do
    if [[ -z "${entry}" ]]; then
      continue
    fi
    runtime_pythonpath="${runtime_pythonpath}${runtime_pythonpath:+:}${entry}"
  done
  local code_pythonpath="${REPO}"
  if [[ -n "${BENCHMARK_PYTHONPATH:-}" ]]; then
    code_pythonpath="${BENCHMARK_PYTHONPATH}:${REPO}"
  fi
  export PYTHONPATH="${code_pythonpath}${venv_pythonpath:+:${venv_pythonpath}}${runtime_pythonpath:+:${runtime_pythonpath}}"
}

write_deploy_config() {
  local policy="${1:-}"
  local enforce_eager="${ENFORCE_EAGER}"
  if [[ "${policy}" == "pr4024_dynamic" ]]; then
    enforce_eager="true"
  fi
  python - "${DEPLOY_CONFIG}" "${DEVICES}" "${TP_SIZE}" "${MAX_NUM_SEQS}" "${enforce_eager}" <<'PY'
import sys

path, devices, tp_size, max_num_seqs, enforce_eager = sys.argv[1:6]
enforce_eager = enforce_eager.strip().lower() not in {"0", "false", "no", "off"}
text = f"""async_chunk: false
trust_remote_code: true

stages:
  - stage_id: 0
    stage_type: diffusion
    max_num_seqs: {int(max_num_seqs)}
    enforce_eager: {str(enforce_eager).lower()}
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
  python - "${policy}" "${raw}" "${stagepool_raw}" "${COST_MODEL}" "${CONSTANT_STEP_MS}" \
    "${STAGEPOOL_LAXITY_WINDOW_MS}" "${STAGEPOOL_PACK_MIN_LAXITY_MS}" \
    "${NO_PREEMPTION_ADMISSION_GUARD}" "${GUARDED_STAGEPOOL_LAXITY_WINDOW_MS}" \
    "${GUARDED_STAGEPOOL_PACK_MAX_QUEUE_LENGTH}" "${GUARDED_STAGEPOOL_PACK_MAX_MATCHING_BUCKET_SIZE}" \
    "${GUARDED_NO_PREEMPTION_ADMISSION_MAX_BATCH_SIZE}" <<'PY'
import json
import sys

(
    policy,
    raw,
    stagepool_raw,
    cost_model,
    constant_step_ms,
    stagepool_laxity_window_ms,
    stagepool_pack_min_laxity_ms,
    no_preemption_admission_guard,
    guarded_stagepool_laxity_window_ms,
    guarded_stagepool_pack_max_queue_length,
    guarded_stagepool_pack_max_matching_bucket_size,
    guarded_no_preemption_admission_max_batch_size,
) = sys.argv[1:13]
dynamic_policies = {
    "pr4024_dynamic",
    "slo_no_preemption_token_guarded",
    "slo_no_preemption_token_adaptive",
    "slo_no_preemption_token_objective",
    "slo_token_stagepool_objective",
    "slo_token_step_preemptive",
    "slo_step_preemptive_token",
    "slo_token_preemptive",
}
token_stagepool_objective_policies = {
    "slo_no_preemption_token_objective",
    "slo_token_stagepool_objective",
}
token_step_preemptive_policies = {
    "slo_token_step_preemptive",
    "slo_step_preemptive_token",
    "slo_token_preemptive",
}
config = {
    "diffusion_dynamic_step_batching_enabled": policy in dynamic_policies,
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
elif policy == "pr4024_dynamic":
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
elif policy == "slo_no_preemption_lookup":
    config["diffusion_scheduler_policy"] = "slo"
    config["diffusion_slo_scheduler"] = {
        **lookup,
        "enable_stagepool_slo": True,
        "enable_step_preemption": False,
        "no_preemption_admission_guard": no_preemption_admission_guard.strip().lower() not in {"0", "false", "no"},
        "stagepool_laxity_window_ms": float(stagepool_laxity_window_ms),
        "stagepool_pack_min_laxity_ms": float(stagepool_pack_min_laxity_ms),
    }
elif policy == "slo_no_preemption_guarded":
    config["diffusion_scheduler_policy"] = "slo"
    config["diffusion_slo_scheduler"] = {
        **lookup,
        "enable_stagepool_slo": True,
        "enable_step_preemption": False,
        "no_preemption_admission_guard": True,
        "stagepool_laxity_window_ms": float(guarded_stagepool_laxity_window_ms),
        "stagepool_pack_min_laxity_ms": float(stagepool_pack_min_laxity_ms),
        "stagepool_pack_max_queue_length": int(float(guarded_stagepool_pack_max_queue_length)),
        "stagepool_pack_max_matching_bucket_size": int(float(guarded_stagepool_pack_max_matching_bucket_size)),
        "no_preemption_admission_max_batch_size": int(float(guarded_no_preemption_admission_max_batch_size)),
    }
elif policy == "slo_no_preemption_token_guarded":
    config["diffusion_scheduler_policy"] = "slo_no_preemption_token_guarded"
    config["diffusion_slo_scheduler"] = {
        **lookup,
        "enable_stagepool_slo": True,
        "enable_step_preemption": False,
        "no_preemption_admission_guard": True,
        "stagepool_laxity_window_ms": float(guarded_stagepool_laxity_window_ms),
        "stagepool_pack_min_laxity_ms": float(stagepool_pack_min_laxity_ms),
        "stagepool_pack_max_queue_length": int(float(guarded_stagepool_pack_max_queue_length)),
        "stagepool_pack_max_matching_bucket_size": int(float(guarded_stagepool_pack_max_matching_bucket_size)),
        "no_preemption_admission_max_batch_size": int(float(guarded_no_preemption_admission_max_batch_size)),
    }
elif policy == "slo_no_preemption_token_adaptive":
    config["diffusion_scheduler_policy"] = "slo_no_preemption_token_adaptive"
    config["diffusion_slo_scheduler"] = {
        **lookup,
        "enable_stagepool_slo": True,
        "enable_step_preemption": False,
        "no_preemption_admission_guard": True,
        "stagepool_laxity_window_ms": float(guarded_stagepool_laxity_window_ms),
        "stagepool_pack_min_laxity_ms": float(stagepool_pack_min_laxity_ms),
        "stagepool_pack_max_queue_length": int(float(guarded_stagepool_pack_max_queue_length)),
        "stagepool_pack_max_matching_bucket_size": int(float(guarded_stagepool_pack_max_matching_bucket_size)),
        "no_preemption_admission_max_batch_size": int(float(guarded_no_preemption_admission_max_batch_size)),
        "adaptive_min_laxity_ratio": 0.10,
        "adaptive_high_token_ratio": 0.75,
        "adaptive_priority_laxity_window_ms": 500.0,
        "adaptive_priority_ratio_window": 0.25,
        "stagepool_profile_reject_laxity_ms": 0.0,
    }
elif policy in token_stagepool_objective_policies:
    config["diffusion_scheduler_policy"] = policy
    config["diffusion_slo_scheduler"] = {
        **lookup,
        "enable_stagepool_slo": True,
        "enable_step_preemption": False,
        "no_preemption_admission_guard": True,
        "stagepool_selection_objective": "token_slo_objective",
        "stagepool_laxity_window_ms": float(guarded_stagepool_laxity_window_ms),
        "stagepool_pack_min_laxity_ms": float(stagepool_pack_min_laxity_ms),
        "stagepool_pack_max_queue_length": int(float(guarded_stagepool_pack_max_queue_length)),
        "stagepool_pack_max_matching_bucket_size": int(float(guarded_stagepool_pack_max_matching_bucket_size)),
        "no_preemption_admission_max_batch_size": int(float(guarded_no_preemption_admission_max_batch_size)),
    }
elif policy in token_step_preemptive_policies:
    config["diffusion_scheduler_policy"] = policy
    config["diffusion_slo_scheduler"] = {
        **lookup,
        "enable_stagepool_slo": True,
        "enable_step_preemption": True,
        "allow_preemptive_admission_swap": True,
        "no_preemption_admission_guard": True,
        "stagepool_laxity_window_ms": float(guarded_stagepool_laxity_window_ms),
        "stagepool_pack_min_laxity_ms": float(stagepool_pack_min_laxity_ms),
        "stagepool_pack_max_queue_length": int(float(guarded_stagepool_pack_max_queue_length)),
        "stagepool_pack_max_matching_bucket_size": int(float(guarded_stagepool_pack_max_matching_bucket_size)),
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

  local additional_config
  additional_config="$(make_additional_config "${policy}" "${raw}" "${stagepool_raw}")"
  local extra_server_args=""
  local extra_server_env=""
  if [[ "${QUANTIZATION_CONFIG}" == *"'"* ]]; then
    echo "QUANTIZATION_CONFIG must not contain a single quote." >&2
    exit 2
  fi
  if [[ -n "${QUANTIZATION_CONFIG}" ]]; then
    extra_server_args="${extra_server_args} --quantization-config '${QUANTIZATION_CONFIG}'"
  fi
  if [[ "${policy}" == "pr4024_dynamic" || "${policy}" == "slo_no_preemption_token_guarded" || "${policy}" == "slo_no_preemption_token_adaptive" || "${policy}" == "slo_no_preemption_token_objective" || "${policy}" == "slo_token_stagepool_objective" || "${policy}" == "slo_token_step_preemptive" || "${policy}" == "slo_step_preemptive_token" || "${policy}" == "slo_token_preemptive" ]]; then
    local dynamic_attention_backend="${PR4024_DYNAMIC_ATTENTION_BACKEND:-FLASH_ATTN}"
    case "${dynamic_attention_backend}" in
      FLASH_ATTN|TORCH_SDPA) ;;
      *)
        echo "Unsupported PR4024_DYNAMIC_ATTENTION_BACKEND=${dynamic_attention_backend}. Expected FLASH_ATTN or TORCH_SDPA." >&2
        exit 2
        ;;
    esac
    extra_server_args="${extra_server_args} --enforce-eager"
    extra_server_env="DIFFUSION_ATTENTION_BACKEND=${dynamic_attention_backend}"
  elif [[ -n "${STATIC_ATTENTION_BACKEND}" ]]; then
    case "${STATIC_ATTENTION_BACKEND}" in
      FLASH_ATTN|TORCH_SDPA) ;;
      *)
        echo "Unsupported STATIC_ATTENTION_BACKEND=${STATIC_ATTENTION_BACKEND}. Expected FLASH_ATTN, TORCH_SDPA, or empty." >&2
        exit 2
        ;;
    esac
    extra_server_env="DIFFUSION_ATTENTION_BACKEND=${STATIC_ATTENTION_BACKEND}"
  fi
  log "starting ${policy}: port=${port}, replicas=${REPLICAS}, tp=${TP_SIZE}, devices=${DEVICES}"
  setsid bash -c "exec env -u DIFFUSION_ATTENTION_BACKEND \
    ASCEND_RT_VISIBLE_DEVICES='${DEVICES}' \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    ${extra_server_env} \
    python -m vllm_omni.entrypoints.cli.main serve '${MODEL}' \
      --omni \
      --host 127.0.0.1 \
      --port '${port}' \
      --tensor-parallel-size '${TP_SIZE}' \
      --max-num-seqs '${MAX_NUM_SEQS}' \
      ${extra_server_args} \
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

clean_policy_result_dirs() {
  local policy="$1"
  local workload
  local interarrival
  for workload in ${WORKLOADS//,/ }; do
    for interarrival in ${INTERARRIVALS//,/ }; do
      rm -rf "${OUTDIR}/${workload}/$(load_label "${interarrival}")/${policy}"
    done
  done
}

make_trace() {
  local policy="$1"
  local repeat="$2"
  local scale="$3"
  local workload="$4"
  local interarrival="$5"
  local label
  local scale_label
  label="$(load_label "${interarrival}")"
  scale_label="$(decimal_label "${scale}")"
  local dir="${OUTDIR}/${workload}/${label}/${policy}/repeat_${repeat}/scale_${scale_label}"
  local trace="${dir}/trace.txt"
  mkdir -p "${dir}"
  log "making trace ${policy} workload=${workload} interarrival=${interarrival} repeat=${repeat} scale=${scale}"
  python benchmarks/diffusion/e2e_slo_first_experiment.py make-trace \
    --output "${trace}" \
    --trace-id "${workload}_${label}_${policy}_r${repeat}_scale_${scale_label}" \
    --profile-mode "${PROFILE_MODE}" \
    --profile-policy "${policy}" \
    --profile-repeat "${repeat}" \
    --profile-load-label "${label}" \
    --workload "${workload}" \
    --num-requests "${NUM_REQUESTS}" \
    --global-interarrival-s "${interarrival}" \
    --slo-scale "${scale}" \
    --step-cost-model-path "${COST_MODEL}" \
    2>&1 | tee -a "${LOG}"
}

make_policy_traces() {
  local policy="$1"
  local workload
  local interarrival
  local repeat
  local scale
  for workload in ${WORKLOADS//,/ }; do
    for interarrival in ${INTERARRIVALS//,/ }; do
      for repeat in ${REPEATS//,/ }; do
        for scale in ${SCALES//,/ }; do
          make_trace "${policy}" "${repeat}" "${scale}" "${workload}" "${interarrival}" || return 1
        done
      done
    done
  done
}

run_benchmark() {
  local policy="$1"
  local repeat="$2"
  local scale="$3"
  local port="$4"
  local workload="$5"
  local interarrival="$6"
  local label
  local scale_label
  label="$(load_label "${interarrival}")"
  scale_label="$(decimal_label "${scale}")"
  local dir="${OUTDIR}/${workload}/${label}/${policy}/repeat_${repeat}/scale_${scale_label}"
  local trace="${dir}/trace.txt"
  local result="${dir}/benchmark_result.json"
  local client_log="${dir}/client.log"
  log "running ${policy} workload=${workload} interarrival=${interarrival} repeat=${repeat} scale=${scale}"
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
  python benchmarks/diffusion/e2e_slo_frontier_experiment.py summarize \
    --output-dir "${OUTDIR}" \
    --policies "${POLICIES}" \
    --candidate-policy "${CANDIDATE_POLICY}" \
    --scales "${SCALES}" \
    --repeats "${REPEATS}" \
    --workloads "${WORKLOADS}" \
    --interarrivals "${INTERARRIVALS}" \
    --profile-mode "${PROFILE_MODE}" \
    > "${OUTDIR}/frontier_summary.stdout.json"
}

policy_results_exist() {
  local policy="$1"
  local workload
  local interarrival
  local repeat
  local scale
  for workload in ${WORKLOADS//,/ }; do
    for interarrival in ${INTERARRIVALS//,/ }; do
      for repeat in ${REPEATS//,/ }; do
        for scale in ${SCALES//,/ }; do
          local result="${OUTDIR}/${workload}/$(load_label "${interarrival}")/${policy}/repeat_${repeat}/scale_$(decimal_label "${scale}")/benchmark_result.json"
          if [[ ! -s "${result}" ]]; then
            return 1
          fi
        done
      done
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
    log "skipping ${policy}: existing results cover workloads=${WORKLOADS}, interarrivals=${INTERARRIVALS}, repeats=${REPEATS}, scales=${SCALES}"
    summarize_results || true
    write_status "skipped_${policy}" "Skipped ${policy}; existing results found"
    return 0
  fi

  write_status "starting_${policy}" "Starting ${policy}"
  clean_policy_result_dirs "${policy}"
  make_policy_traces "${policy}" || return 1
  write_deploy_config "${policy}"
  start_service "${policy}" "${port}" "${master_port}"
  ACTIVE_PID_FILE="${pid_file}"
  wait_health "${port}" || { stop_service "${pid_file}"; ACTIVE_PID_FILE=""; return 1; }
  local workload
  local interarrival
  local repeat
  local scale
  for workload in ${WORKLOADS//,/ }; do
    for interarrival in ${INTERARRIVALS//,/ }; do
      for repeat in ${REPEATS//,/ }; do
        for scale in ${SCALES//,/ }; do
          run_benchmark "${policy}" "${repeat}" "${scale}" "${port}" "${workload}" "${interarrival}" || {
            stop_service "${pid_file}"
            ACTIVE_PID_FILE=""
            return 1
          }
          summarize_results || true
          write_status "running_${policy}" "Completed ${policy} workload=${workload} interarrival=${interarrival} repeat=${repeat} scale=${scale}"
        done
      done
    done
  done
  stop_service "${pid_file}"
  ACTIVE_PID_FILE=""
  write_status "completed_${policy}" "Completed ${policy}"
}

main() {
  validate_matrix_args
  setup_env
  log "output dir: ${OUTDIR}"
  log "config: model=${MODEL}, replicas=${REPLICAS}, tp=${TP_SIZE}, requests=${NUM_REQUESTS}, interarrivals=${INTERARRIVALS}, workloads=${WORKLOADS}, scales=${SCALES}, repeats=${REPEATS}, policies=${POLICIES}, candidate=${CANDIDATE_POLICY}"
  local policy
  for policy in ${POLICIES//,/ }; do
    run_policy "${policy}" || { write_status failed "${policy} failed"; exit 1; }
  done
  summarize_results
  write_status completed "E2E SLO frontier completed"
  log "completed: ${OUTDIR}"
}

main "$@"
