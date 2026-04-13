#!/bin/bash
# Start an OpenAI-compatible vLLM-Omni server for Wan2.2 S2V.
#
# Environment variables let local smoke tests override the model path, port,
# and startup flags without editing the script itself.

set -euo pipefail

MODEL="${MODEL:-/path/to/Wan-AI--Wan2.2-S2V-14B}"
PORT="${PORT:-8099}"
ENABLE_CPU_OFFLOAD="${ENABLE_CPU_OFFLOAD:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
DISABLE_LOG_STATS="${DISABLE_LOG_STATS:-1}"

cmd=(vllm serve "$MODEL" --omni --port "$PORT")

if [ "$ENABLE_CPU_OFFLOAD" != "0" ]; then
  cmd+=(--enable-cpu-offload)
fi
if [ "$ENFORCE_EAGER" != "0" ]; then
  cmd+=(--enforce-eager)
fi
if [ "$DISABLE_LOG_STATS" != "0" ]; then
  cmd+=(--disable-log-stats)
fi

echo "Starting Wan2.2 S2V server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Enable CPU offload: $ENABLE_CPU_OFFLOAD"
echo "Enforce eager: $ENFORCE_EAGER"

"${cmd[@]}"
