# Wan2.2 S2V

This example shows how to serve `Wan2.2-S2V-14B` through vLLM-Omni's OpenAI-compatible
`/v1/videos` API for both audio-driven and TTS-driven speech-to-video generation.

## Start Server

```bash
bash run_server.sh
```

Environment variables supported by the script:

- `MODEL`: local Wan2.2 S2V checkpoint root.
- `PORT`: API server port. Default: `8099`.
- `ENABLE_CPU_OFFLOAD`: pass `--enable-cpu-offload`. Default: `1`.
- `ENFORCE_EAGER`: pass `--enforce-eager`. Default: `0`.
- `DISABLE_LOG_STATS`: pass `--disable-log-stats`. Default: `1`.

`Wan2.2-S2V-14B` currently requires a local Wan root-layout checkpoint directory and
expects the upstream `wan` Python package to be installed separately in the same environment.

## Async Job Behavior

`POST /v1/videos` is asynchronous. It creates a video job and immediately returns
metadata such as the job ID and initial `queued` status. Poll the job until it is
`completed`, then download the generated MP4 from the content endpoint.

The main endpoints are:

- `POST /v1/videos`: create a video generation job (async)
- `POST /v1/videos/sync`: generate a video and return raw bytes (sync, for smoke and benchmarks)
- `GET /v1/videos/{video_id}`: retrieve the current job status and metadata
- `GET /v1/videos`: list stored video jobs
- `GET /v1/videos/{video_id}/content`: download the generated video file
- `DELETE /v1/videos/{video_id}`: delete the job and any stored output

## Sync API

`POST /v1/videos/sync` blocks until generation completes and returns the raw MP4 bytes.
It is useful for smoke tests and latency measurements.

Metadata is returned through response headers:

- `X-Request-Id`
- `X-Model`
- `X-Inference-Time-S`

Audio-driven sync request example:

```bash
curl -X POST http://localhost:8099/v1/videos/sync \
  -F "prompt=A person is talking naturally to the camera." \
  -F "input_reference=@/path/to/i2v_input.JPG" \
  -F "input_audio_reference=@/path/to/talk.wav" \
  -F "width=832" \
  -F "height=448" \
  -F "num_frames=17" \
  -F "num_inference_steps=2" \
  -F "guidance_scale=4.5" \
  -F "seed=42" \
  -F 'extra_params={"infer_frames":16,"num_repeat":1,"shift":3.0,"solver_name":"unipc","offload_model":true}' \
  -o wan22_s2v_sync.mp4
```

For TTS-driven sync requests, use the same endpoint with `enable_tts=true`,
`input_tts_prompt_audio` or `tts_prompt_audio`, `tts_text`, and optional
`tts_prompt_text`.

## Storage

Generated video files are stored on local disk by the async video API.
You can control storage behavior with the following environment variables:

- `VLLM_OMNI_STORAGE_PATH`: directory used for generated files (default: `/tmp/storage`)
- `VLLM_OMNI_STORAGE_MAX_CONCURRENCY`: max concurrent save/delete operations (default: `4`)

Example:

```bash
export VLLM_OMNI_STORAGE_PATH=/var/tmp/vllm-omni-videos
export VLLM_OMNI_STORAGE_MAX_CONCURRENCY=8
```

## Audio-Driven S2V

Use the helper script:

```bash
bash run_curl_s2v.sh
```

Equivalent raw request:

```bash
curl -X POST http://localhost:8099/v1/videos \
  -H "Accept: application/json" \
  -F "prompt=A person is talking naturally to the camera." \
  -F "input_reference=@/path/to/i2v_input.JPG" \
  -F "input_audio_reference=@/path/to/talk.wav" \
  -F "width=832" \
  -F "height=448" \
  -F "num_frames=17" \
  -F "num_inference_steps=2" \
  -F "guidance_scale=4.5" \
  -F "seed=42" \
  -F 'extra_params={"infer_frames":16,"num_repeat":1,"shift":3.0,"solver_name":"unipc","offload_model":true}'
```

The driving audio can be provided in either of these forms:

- `input_audio_reference`: uploaded audio file
- `audio_reference`: `http(s)` URL, `data:` URL, or an allowed local media URL

Do not provide both in the same request.

## TTS-Driven S2V

Use the helper script:

```bash
bash run_curl_s2v_tts.sh
```

Equivalent raw request:

```bash
curl -X POST http://localhost:8099/v1/videos \
  -H "Accept: application/json" \
  -F "prompt=A cheerful person is speaking to the camera." \
  -F "input_reference=@/path/to/i2v_input.JPG" \
  -F "enable_tts=true" \
  -F "input_tts_prompt_audio=@/path/to/zero_shot_prompt.wav" \
  -F "tts_prompt_text=希望你以后能够做的比我还好呦。" \
  -F "tts_text=收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。" \
  -F "width=832" \
  -F "height=448" \
  -F "num_frames=17" \
  -F "num_inference_steps=2" \
  -F "guidance_scale=4.5" \
  -F "seed=42" \
  -F 'extra_params={"infer_frames":16,"num_repeat":1,"shift":3.0,"solver_name":"unipc","offload_model":true}'
```

The TTS prompt audio can be provided in either of these forms:

- `input_tts_prompt_audio`: uploaded audio file
- `tts_prompt_audio`: `http(s)` URL, `data:` URL, or an allowed local media URL

Do not provide both in the same request.

## Request Notes

- Audio-driven mode requires an input image and a driving audio source.
- TTS-driven mode requires an input image, `enable_tts=true`, a TTS prompt audio source,
  and a non-empty `tts_text`.
- `tts_prompt_text` is optional.
- Model-specific runtime controls currently go through `extra_params`, especially
  `infer_frames`, `num_repeat`, `shift`, `solver_name`, and `offload_model`.
- For non-TTS S2V, vLLM-Omni muxes the driving audio back into the final MP4 when the
  model itself does not return a dedicated audio track.

## Create Response Format

`POST /v1/videos` returns a job record, not inline base64 video data.

```json
{
  "id": "video_gen_123",
  "object": "video",
  "status": "queued",
  "model": "/path/to/Wan-AI--Wan2.2-S2V-14B",
  "prompt": "A person is talking naturally to the camera.",
  "created_at": 1234567890
}
```

## Retrieve, List, Download, and Delete

```bash
curl -s http://localhost:8099/v1/videos/${video_id} | jq .
curl -s http://localhost:8099/v1/videos | jq .
curl -L http://localhost:8099/v1/videos/${video_id}/content -o wan22_s2v_output.mp4
curl -X DELETE http://localhost:8099/v1/videos/${video_id} | jq .
```
