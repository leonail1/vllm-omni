# Diffusion Step Cost Profile Report

- raw: `/home/lzg/vllm-omni/benchmark_outputs/latent_token_aspect_profile_20260522_215548/step_cost_raw.jsonl`
- table rows: `92`
- fallback available: `True`

## Repeat Stability

- `1024x1024_b1` repeats=3, median_spread=0.02%, p90_spread=0.04%, stable=True
- `1024x1024_b2` repeats=3, median_spread=0.23%, p90_spread=0.25%, stable=True
- `1024x1024_b3` repeats=3, median_spread=0.24%, p90_spread=0.24%, stable=True
- `1024x1024_b4` repeats=3, median_spread=0.02%, p90_spread=0.03%, stable=True
- `1024x256_b1` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `1024x256_b2` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `1024x256_b3` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `1024x256_b4` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `1024x400_b1` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `1024x400_b2` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `1024x400_b3` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `1024x400_b4` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `1024x576_b1` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `1024x576_b2` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `1024x576_b3` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `1024x576_b4` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `1024x784_b1` repeats=3, median_spread=9.38%, p90_spread=11.07%, stable=False
- `1024x784_b2` repeats=3, median_spread=0.26%, p90_spread=0.29%, stable=True
- `1024x784_b3` repeats=3, median_spread=0.60%, p90_spread=0.61%, stable=True
- `1024x784_b4` repeats=3, median_spread=0.17%, p90_spread=0.17%, stable=True
- `1152x512_b1` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `1152x512_b2` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `1152x512_b3` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `1152x512_b4` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `1792x448_b1` repeats=3, median_spread=6.22%, p90_spread=7.99%, stable=False
- `1792x448_b2` repeats=3, median_spread=0.13%, p90_spread=0.05%, stable=True
- `1792x448_b3` repeats=3, median_spread=0.28%, p90_spread=0.29%, stable=True
- `1792x448_b4` repeats=3, median_spread=0.22%, p90_spread=0.24%, stable=True
- `2048x512_b1` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `2048x512_b2` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `2048x512_b3` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `2048x512_b4` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `256x1024_b1` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `256x1024_b2` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `256x1024_b3` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `256x1024_b4` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `256x256_b1` repeats=3, median_spread=2.49%, p90_spread=1.07%, stable=True
- `256x256_b2` repeats=3, median_spread=1.82%, p90_spread=4.16%, stable=True
- `256x256_b3` repeats=3, median_spread=20.05%, p90_spread=18.60%, stable=False
- `256x256_b4` repeats=3, median_spread=1.27%, p90_spread=0.78%, stable=True
- `384x384_b1` repeats=3, median_spread=0.88%, p90_spread=6.44%, stable=True
- `384x384_b2` repeats=3, median_spread=1.29%, p90_spread=2.60%, stable=True
- `384x384_b3` repeats=3, median_spread=3.50%, p90_spread=3.94%, stable=True
- `384x384_b4` repeats=3, median_spread=0.93%, p90_spread=1.80%, stable=True
- `400x1024_b1` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `400x1024_b2` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `400x1024_b3` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `400x1024_b4` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `448x1792_b1` repeats=3, median_spread=0.15%, p90_spread=0.22%, stable=True
- `448x1792_b2` repeats=3, median_spread=0.15%, p90_spread=0.12%, stable=True
- `448x1792_b3` repeats=3, median_spread=0.53%, p90_spread=0.59%, stable=True
- `448x1792_b4` repeats=3, median_spread=0.01%, p90_spread=0.08%, stable=True
- `512x1152_b1` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `512x1152_b2` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `512x1152_b3` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `512x1152_b4` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `512x2048_b1` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `512x2048_b2` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `512x2048_b3` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `512x2048_b4` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `512x512_b1` repeats=3, median_spread=2.62%, p90_spread=2.36%, stable=True
- `512x512_b2` repeats=3, median_spread=0.91%, p90_spread=0.64%, stable=True
- `512x512_b3` repeats=3, median_spread=0.03%, p90_spread=0.06%, stable=True
- `512x512_b4` repeats=3, median_spread=0.12%, p90_spread=0.48%, stable=True
- `512x800_b1` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `512x800_b2` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `512x800_b3` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `512x800_b4` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `576x1024_b1` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `576x1024_b2` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `576x1024_b3` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `576x1024_b4` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `640x640_b1` repeats=3, median_spread=3.37%, p90_spread=4.07%, stable=True
- `640x640_b2` repeats=3, median_spread=0.07%, p90_spread=0.08%, stable=True
- `640x640_b3` repeats=3, median_spread=0.18%, p90_spread=0.08%, stable=True
- `640x640_b4` repeats=3, median_spread=0.08%, p90_spread=0.32%, stable=True
- `768x768_b1` repeats=3, median_spread=1.20%, p90_spread=0.83%, stable=True
- `768x768_b2` repeats=3, median_spread=0.47%, p90_spread=0.63%, stable=True
- `768x768_b3` repeats=3, median_spread=0.10%, p90_spread=0.34%, stable=True
- `768x768_b4` repeats=3, median_spread=0.08%, p90_spread=0.24%, stable=True
- `784x1024_b1` repeats=3, median_spread=0.19%, p90_spread=0.43%, stable=True
- `784x1024_b2` repeats=3, median_spread=0.07%, p90_spread=0.06%, stable=True
- `784x1024_b3` repeats=3, median_spread=0.54%, p90_spread=0.53%, stable=True
- `784x1024_b4` repeats=3, median_spread=0.29%, p90_spread=0.29%, stable=True
- `800x512_b1` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `800x512_b2` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `800x512_b3` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `800x512_b4` repeats=1, median_spread=0.00%, p90_spread=0.00%, stable=True
- `896x896_b1` repeats=3, median_spread=0.06%, p90_spread=0.15%, stable=True
- `896x896_b2` repeats=3, median_spread=153.62%, p90_spread=594.44%, stable=False
- `896x896_b3` repeats=3, median_spread=1.88%, p90_spread=14.66%, stable=False
- `896x896_b4` repeats=3, median_spread=0.29%, p90_spread=0.24%, stable=True

## Step Index Stability

- available: `True`
- max step median deviation: `1.48%`
- use step index in model: `False`

## Mixed Step Sanity

- No mixed-step sanity rows found.

## Failures

- No request failures found.

## Scheduler Recommendation

- Collapse step index and use `denoise_step_ms = f(shape, batch_size, effective_batch_size)`.
