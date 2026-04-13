#!/bin/bash
# Submit a TTS-driven Wan2.2 S2V request to the async video API.
#
# The script uploads a reference image plus TTS prompt audio, waits for the
# async job to finish, and downloads the generated MP4 output.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="${BASE_URL:-http://localhost:8099}"
INPUT_IMAGE="${INPUT_IMAGE:-$SCRIPT_DIR/../../../upstream_refs/Wan2.2/examples/i2v_input.JPG}"
PROMPT_AUDIO="${PROMPT_AUDIO:-$SCRIPT_DIR/../../../upstream_refs/Wan2.2/examples/zero_shot_prompt.wav}"
PROMPT_TEXT="${PROMPT_TEXT:-希望你以后能够做的比我还好呦。}"
TTS_TEXT="${TTS_TEXT:-收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。}"
OUTPUT_PATH="${OUTPUT_PATH:-wan22_s2v_tts_output.mp4}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"

if [ ! -f "$INPUT_IMAGE" ]; then
  echo "Input image not found: $INPUT_IMAGE"
  exit 1
fi
if [ ! -f "$PROMPT_AUDIO" ]; then
  echo "Prompt audio not found: $PROMPT_AUDIO"
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required for this script."
  exit 1
fi

create_response=$(curl -sS -X POST "${BASE_URL}/v1/videos" \
  -H "Accept: application/json" \
  -F "prompt=A cheerful person is speaking to the camera." \
  -F "input_reference=@${INPUT_IMAGE}" \
  -F "enable_tts=true" \
  -F "input_tts_prompt_audio=@${PROMPT_AUDIO}" \
  -F "tts_prompt_text=${PROMPT_TEXT}" \
  -F "tts_text=${TTS_TEXT}" \
  -F "width=832" \
  -F "height=448" \
  -F "num_frames=17" \
  -F "num_inference_steps=2" \
  -F "guidance_scale=4.5" \
  -F "seed=42" \
  -F 'extra_params={"infer_frames":16,"num_repeat":1,"shift":3.0,"solver_name":"unipc","offload_model":true}')

video_id="$(echo "$create_response" | jq -r '.id')"
if [ -z "$video_id" ] || [ "$video_id" = "null" ]; then
  echo "Failed to create video job:"
  echo "$create_response" | jq .
  exit 1
fi

echo "Created video job $video_id"
echo "$create_response" | jq .

while true; do
  status_response="$(curl -sS "${BASE_URL}/v1/videos/${video_id}")"
  status="$(echo "$status_response" | jq -r '.status')"
  case "$status" in
    queued|in_progress)
      echo "Video job $video_id status: $status"
      sleep "$POLL_INTERVAL"
      ;;
    completed)
      echo "$status_response" | jq .
      break
      ;;
    failed)
      echo "Video generation failed:"
      echo "$status_response" | jq .
      exit 1
      ;;
    *)
      echo "Unexpected status response:"
      echo "$status_response" | jq .
      exit 1
      ;;
  esac
done

curl -sS -L "${BASE_URL}/v1/videos/${video_id}/content" -o "$OUTPUT_PATH"
echo "Saved video to $OUTPUT_PATH"
