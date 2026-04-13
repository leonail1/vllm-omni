# Wan2.2 S2V Offline Inference

This example runs `Wan-AI/Wan2.2-S2V-14B` through the vLLM-Omni `WanS2VPipeline`
for speech-driven video generation.

## Requirements

- Download `Wan-AI/Wan2.2-S2V-14B` to a local Wan root-layout directory.
- Install the upstream `wan` Python package into the same environment as `vllm-omni`.
  The runtime no longer imports code from `upstream_refs/Wan2.2` directly.
- `upstream_refs/Wan2.2` is only used here for the bundled example assets in this repository.
- Install `vllm-omni`, the native `Wan2.2` inference dependencies, and `ffmpeg`
  if you want `--merge-audio`.
- For `--enable-tts`, also install the extra dependencies from the upstream Wan2.2
  repo's `requirements_s2v.txt`.

## Audio-Driven S2V

```bash
python examples/offline_inference/wan2_2_s2v/end2end.py \
  --model /mnt/data/lzg/model_weights/Wan-AI--Wan2.2-S2V-14B \
  --prompt "A person is speaking to the camera in a bright room." \
  --image upstream_refs/Wan2.2/examples/i2v_input.JPG \
  --audio upstream_refs/Wan2.2/examples/talk.wav \
  --height 448 \
  --width 832 \
  --num-frames 17 \
  --infer-frames 16 \
  --num-inference-steps 8 \
  --guidance-scale 4.5 \
  --shift 3.0 \
  --enforce-eager \
  --enable-cpu-offload \
  --merge-audio \
  --output wan22_s2v_demo.mp4
```

## TTS-Driven S2V

```bash
python examples/offline_inference/wan2_2_s2v/end2end.py \
  --model /mnt/data/lzg/model_weights/Wan-AI--Wan2.2-S2V-14B \
  --prompt "A cheerful person is speaking to the camera." \
  --image upstream_refs/Wan2.2/examples/i2v_input.JPG \
  --enable-tts \
  --tts-prompt-audio upstream_refs/Wan2.2/examples/zero_shot_prompt.wav \
  --tts-prompt-text "希望你以后能够做的比我还好呦。" \
  --tts-text "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。" \
  --height 448 \
  --width 832 \
  --num-frames 17 \
  --infer-frames 16 \
  --num-inference-steps 8 \
  --guidance-scale 4.5 \
  --shift 3.0 \
  --enforce-eager \
  --enable-cpu-offload \
  --merge-audio \
  --output wan22_s2v_tts_demo.mp4
```

## Key Arguments

- `--num-frames`: requested output length on the vLLM-Omni side. Wan S2V expects `4n+1`.
- `--infer-frames`: native Wan per-clip generation length. Wan S2V expects `4n`.
- `--num-repeat`: maximum number of generated clips. The backend can still reduce it when the
  driving audio is shorter than requested.
- `--pose-video`: optional pose-condition input reused from the native Wan S2V flow.
- `--enable-tts`: synthesize the driving speech with CosyVoice before S2V generation. In this
  mode `--audio` is optional.
- `--tts-prompt-audio` and `--tts-text`: required when TTS mode is enabled.
- `--tts-prompt-text`: optional transcript for the TTS prompt audio.
- `--height` / `--width`: use 64-aligned sizes. `448x832` matches the current smoke-test setup.
- `--enforce-eager`: useful for local smoke tests because it skips CUDA graph capture and reduces
  first-run initialization overhead.

## Notes

- The current implementation supports local Wan root-layout checkpoints.
- Both `image + audio` mode and `image + TTS` mode are supported, with optional `pose_video`.
- The CosyVoice TTS assets are cached under a sibling `wan22_s2v_tts_assets/` directory near the model path.
- `--merge-audio` works for both driving-audio mode and TTS mode. When the pipeline produces a
  synthesized WAV, that audio is muxed back into the final MP4 automatically.
