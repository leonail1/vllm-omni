# Wan2.2 S2V Online Serving

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/wan2_2_s2v>.

This example demonstrates how to serve `Wan2.2-S2V-14B` through vLLM-Omni's OpenAI-compatible
video API for both audio-driven and TTS-driven speech-to-video generation.

## Start Server

```bash
bash examples/online_serving/wan2_2_s2v/run_server.sh
```

`Wan2.2-S2V-14B` currently expects a local Wan root-layout checkpoint path and
requires the upstream `wan` Python package to be installed separately in the same environment.

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

## Audio-Driven S2V

```bash
bash examples/online_serving/wan2_2_s2v/run_curl_s2v.sh
```

The API accepts the driving audio either as:

- `input_audio_reference`: uploaded audio file
- `audio_reference`: `http(s)` URL, `data:` URL, or allowed local media URL

Do not provide both in the same request.

## TTS-Driven S2V

```bash
bash examples/online_serving/wan2_2_s2v/run_curl_s2v_tts.sh
```

The API accepts the TTS prompt audio either as:

- `input_tts_prompt_audio`: uploaded audio file
- `tts_prompt_audio`: `http(s)` URL, `data:` URL, or allowed local media URL

Do not provide both in the same request.

TTS-specific request fields:

- `enable_tts`
- `tts_prompt_text`
- `tts_text`

## Request Notes

- Audio-driven mode requires an input image and a driving audio source.
- TTS-driven mode requires an input image, `enable_tts=true`, a TTS prompt audio source,
  and a non-empty `tts_text`.
- Model-specific runtime controls currently go through `extra_params`, especially
  `infer_frames`, `num_repeat`, `shift`, `solver_name`, and `offload_model`.
- For non-TTS S2V, vLLM-Omni muxes the driving audio back into the final MP4 when the
  model itself does not return a dedicated audio track.

## Example Materials

??? abstract "README.md"
    ``````md
    --8<-- "examples/online_serving/wan2_2_s2v/README.md"
    ``````
??? abstract "run_server.sh"
    ``````bash
    --8<-- "examples/online_serving/wan2_2_s2v/run_server.sh"
    ``````
??? abstract "run_curl_s2v.sh"
    ``````bash
    --8<-- "examples/online_serving/wan2_2_s2v/run_curl_s2v.sh"
    ``````
??? abstract "run_curl_s2v_tts.sh"
    ``````bash
    --8<-- "examples/online_serving/wan2_2_s2v/run_curl_s2v_tts.sh"
    ``````
