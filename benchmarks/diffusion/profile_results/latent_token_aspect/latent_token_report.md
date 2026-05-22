# Latent Tokens vs Aspect Ratio Profile Report

- raw: `/home/lzg/vllm-omni/benchmark_outputs/latent_token_aspect_profile_20260522_215548/step_cost_raw.jsonl`
- aspect rows: `92`
- preferred_key: `shape_key`
- fallback_features: `['latent_tokens', 'effective_batch_size', 'aspect_ratio']`

## 关键结论

- 512x2048 / 2048x512 与 1024x1024 接近。
- 同 latent_tokens 下，横图和竖图的最大 median 差异为 `4.21%`，可以认为基本对称。
- aspect_ratio 需要进入 cost model。
- 第一版 scheduler 建议继续使用 `shape_key` 作为 table lookup 主键，fallback 加入 `aspect_ratio`。

## 1024x1024 等 token 矩形

- `512x2048x1` batch=1: median_delta=-0.04%, p95_delta=-0.05%
- `2048x512x1` batch=1: median_delta=-0.02%, p95_delta=0.12%
- `512x2048x1` batch=2: median_delta=-0.31%, p95_delta=-0.40%
- `2048x512x1` batch=2: median_delta=-0.06%, p95_delta=-0.08%
- `512x2048x1` batch=3: median_delta=0.09%, p95_delta=-0.11%
- `2048x512x1` batch=3: median_delta=0.04%, p95_delta=-0.16%
- `512x2048x1` batch=4: median_delta=0.05%, p95_delta=0.03%
- `2048x512x1` batch=4: median_delta=0.05%, p95_delta=0.02%

## 模型对比

- in-sample: tokens-only MAPE=6.29%, tokens+aspect MAPE=5.59%, improvement=0.70 pct-points.
- leave-one-shape-out: tokens-only MAPE=7.06%, tokens+aspect MAPE=7.19%, improvement=-0.13 pct-points.

## Risk Shapes

- `1024x784x1` batch=1, aspect=1.31, median_delta=0.30%, p95_delta=10.29%
- `1024x784x1` batch=2, aspect=1.31, median_delta=-0.12%, p95_delta=-79.60%
- `1024x784x1` batch=3, aspect=1.31, median_delta=-0.34%, p95_delta=-12.11%
- `1792x448x1` batch=2, aspect=4.00, median_delta=-0.09%, p95_delta=-79.61%
- `1792x448x1` batch=3, aspect=4.00, median_delta=-0.06%, p95_delta=-12.31%
- `448x1792x1` batch=2, aspect=4.00, median_delta=-0.18%, p95_delta=-79.63%
- `448x1792x1` batch=3, aspect=4.00, median_delta=-0.09%, p95_delta=-12.08%
- `784x1024x1` batch=2, aspect=1.31, median_delta=-0.08%, p95_delta=-79.61%
- `784x1024x1` batch=3, aspect=1.31, median_delta=-0.04%, p95_delta=-12.12%
- `800x512x1` batch=1, aspect=1.56, median_delta=4.87%, p95_delta=15.54%

## Failures

- No request failures found.
