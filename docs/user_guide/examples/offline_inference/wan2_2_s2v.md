# Wan2.2 S2V Offline Inference

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/wan2_2_s2v>.

This example demonstrates how to run `Wan2.2-S2V-14B` with vLLM-Omni's offline inference API.

## Local CLI Usage

Audio-driven S2V:

```bash
python examples/offline_inference/wan2_2_s2v/end2end.py \
  --model /path/to/Wan-AI--Wan2.2-S2V-14B \
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

TTS-driven S2V:

```bash
python examples/offline_inference/wan2_2_s2v/end2end.py \
  --model /path/to/Wan-AI--Wan2.2-S2V-14B \
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
  --output wan22_s2v_tts_demo.mp4
```

## Notes

- `--audio` is required in audio-driven mode.
- `--enable-tts` switches the pipeline to CosyVoice-driven speech synthesis before S2V generation.
- `--tts-prompt-audio` and `--tts-text` are required in TTS mode; `--tts-prompt-text` is optional.
- `--merge-audio` muxes the driving or synthesized audio back into the final MP4 when `ffmpeg` is available.

## Example Materials

??? abstract "README.md"
    ``````md
    --8<-- "examples/offline_inference/wan2_2_s2v/README.md"
    ``````
??? abstract "end2end.py"
    ``````py
    --8<-- "examples/offline_inference/wan2_2_s2v/end2end.py"
    ``````
