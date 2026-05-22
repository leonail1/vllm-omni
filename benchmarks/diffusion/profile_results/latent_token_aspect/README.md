# Qwen-Image Latent Tokens vs. Aspect Ratio Step Cost Profile

本目录保留 910B 上一次可复现的 Qwen-Image denoise step cost 实验，包括原始 raw、聚合结果、分析报告和脚本副本。

## 实验信息

- 模型：`Qwen/Qwen-Image`
- 机器：910B
- 设备：`ASCEND_RT_VISIBLE_DEVICES=0,1`
- 并行：`tensor_parallel_size=2`
- 服务：单个 diffusion instance，`--omni-dp-size-local 1`
- 调度：`--step-execution`
- `max_num_seqs=4`
- `num_inference_steps=50`
- CFG：关闭，`num_outputs_per_prompt=1`
- 原始输出目录：`/home/lzg/vllm-omni/benchmark_outputs/latent_token_aspect_profile_20260522_215548`

## 目录内容

- `step_cost_raw.jsonl`：每一次真实 bucket step 的原始 profile 记录。
- `profile_run_manifest.jsonl`：请求级 manifest，用于确认每个组合是否成功。
- `aspect_equivalence_table.csv`：同 `latent_tokens + batch_size` 下，不同 shape 的 median/p90/p95 和相对 square baseline 的 delta。
- `step_cost_profile_table.csv`：按 `model, shape_key, batch_size, effective_batch_size` 聚合的 step cost 表。
- `step_cost_model.json`：profile 工具生成的 table lookup + fallback 模型。
- `model_comparison.json`：tokens-only 与 tokens+aspect 模型的拟合/交叉验证对比。
- `latent_token_report.md`：自动生成的结论报告。
- `profile_report.md`、`profile_diagnostics.json`：repeat 稳定性、step index、失败请求等诊断。
- `full_experiment/`：runner/server 日志和最终状态。
- `scripts/run_910b_latent_token_aspect_profile.sh`：启动服务、跑矩阵和聚合结果的脚本副本。
- `scripts/step_cost_profile.py`：发送 profile 流量、聚合和分析结果的工具副本。

## 实验矩阵

同 latent tokens、不同长宽比：

```text
latent_tokens=1024: 512x512, 256x1024, 1024x256
latent_tokens=1600: 640x640, 512x800, 800x512, 400x1024, 1024x400
latent_tokens=2304: 768x768, 512x1152, 1152x512, 576x1024, 1024x576
latent_tokens=3136: 896x896, 784x1024, 1024x784, 448x1792, 1792x448
latent_tokens=4096: 1024x1024, 512x2048, 2048x512
```

用于拟合 token 规模趋势的额外 square shape：

```text
256x256, 384x384, 512x512, 640x640, 768x768, 896x896, 1024x1024
```

batch size：

```text
1, 2, 3, 4
```

repeat 策略：

- 每个组合 1 次 warmup，不计入结果。
- square shape 默认 3 repeats。
- rectangle shape 默认 1 repeat。
- 若 square repeat 不稳定，则对应 token group 的 rectangle 补到 3 repeats。
- 本次触发补跑的 token group：`256`、`3136`。其中 `256` 没有 rectangle shape，因此实际补跑主要发生在 `3136` 的矩形组合。

## Latent Tokens 定义

Qwen-Image 的 latent token 数按以下公式计算：

```text
latent_tokens = (width / 16) * (height / 16)
```

其中 `width` 和 `height` 都需要能被 16 整除。为了让不同 token 规模可以在一个线性模型里比较，拟合时使用归一化变量：

```text
T = latent_tokens / 4096
B = effective_batch_size
```

本实验关闭 CFG，`num_outputs_per_prompt=1`，因此：

```text
effective_batch_size = batch_size
```

## 核心结果

`512x2048 / 2048x512` 与 `1024x1024` 的差异非常小：

```text
batch=1:
  512x2048 median_delta=-0.04%, p95_delta=-0.05%
  2048x512 median_delta=-0.02%, p95_delta=0.12%

batch=2:
  512x2048 median_delta=-0.31%, p95_delta=-0.40%
  2048x512 median_delta=-0.06%, p95_delta=-0.08%

batch=3:
  512x2048 median_delta=0.09%, p95_delta=-0.11%
  2048x512 median_delta=0.04%, p95_delta=-0.16%

batch=4:
  512x2048 median_delta=0.05%, p95_delta=0.03%
  2048x512 median_delta=0.05%, p95_delta=0.02%
```

各 token group 的最大 median delta：

```text
tokens=1024: max_abs_median_delta=2.92%
tokens=1600: max_abs_median_delta=4.87%
tokens=2304: max_abs_median_delta=4.61%
tokens=3136: max_abs_median_delta=0.34%
tokens=4096: max_abs_median_delta=0.31%
```

横图/竖图对称性：

```text
max pairwise median delta = 4.21%
```

因此，按 median step cost 看，第一版 scheduler 可以将分辨率压缩成：

```text
latent_tokens + effective_batch_size
```

不需要把 `aspect_ratio` 作为主特征。

## p95 Risk Shapes 的解释

自动报告中 `aspect_needed=true`，原因是它按规则把 `p95_delta` 超过阈值的行标记为 risk shape。典型风险行来自 `tokens=3136`：

```text
1024x784 / 784x1024 / 448x1792 / 1792x448
```

这些行的 median delta 接近 0，但 p95 delta 出现很大偏差，说明问题更像 square baseline 的 tail/outlier 或 repeat 抖动，而不是矩形长宽比稳定改变了 denoise cost。交叉验证也支持这一点：

```text
tokens-only leave-one-shape-out MAPE = 7.06%
tokens+aspect leave-one-shape-out MAPE = 7.19%
```

加入 aspect 后没有提升泛化效果。

## 公式推导

DiT denoise step 的主要计算与 latent token 数和 batch 内样本数有关。对于同一个模型、同一硬件、同一并行度，可以先把每次 bucket step 的 cost 近似成：

```text
step_ms = f(latent_tokens, effective_batch_size)
```

用归一化 token 数 `T = latent_tokens / 4096`、等效 batch size `B` 表示，最基础的线性交互模型是：

```text
step_ms = c0 + c1*T + c2*B + c3*T*B
```

各项含义：

- `c0`：固定开销，包括 scheduler tick、运行时调度、同步、kernel launch 等。
- `c1*T`：单样本 token 规模带来的基础开销。
- `c2*B`：每增加一个样本带来的 batch 级固定开销。
- `c3*T*B`：核心计算项，近似表示 batch 中总 token 工作量。

但 Qwen-Image 的 DiT block 中存在 attention/MLP 等非线性计算和硬件效率变化，仅靠 `T*B` 对大 token 区间的拟合不够稳定。因此加入 token 二次交互项：

```text
step_ms = c0 + c1*T + c2*B + c3*T*B + c4*T^2*B
```

直觉是：当 latent tokens 增大时，attention 和缓存/带宽压力的增长并不完全线性，`T^2*B` 用来吸收这部分随 token 规模加速增长的 cost。

## 拟合结果

全量数据，`n=92`：

```text
step_ms ≈ 319.25
        - 273.16*T
        - 17.38*B
        + 246.55*T*B
        + 158.92*T^2*B

MAPE = 3.93%
p90 APE = 10.85%
max APE = 15.69%
```

只看 `latent_tokens >= 1024` 的生产相关尺寸，`n=84`：

```text
step_ms ≈ 288.68
        - 230.01*T
        - 38.12*B
        + 335.52*T*B
        + 82.63*T^2*B

MAPE = 2.65%
p90 APE = 7.61%
max APE = 13.57%
```

对比模型：

```text
1 + T + B + T*B:
  all MAPE = 6.29%
  tokens>=1024 MAPE = 3.38%

1 + T + B + T*B + T*B^2:
  all MAPE = 6.33%
  tokens>=1024 MAPE = 3.22%

1 + T + B + T*B + T^2*B:
  all MAPE = 3.93%
  tokens>=1024 MAPE = 2.65%
```

因此推荐第一版 fallback 公式：

```text
step_ms = c0 + c1*T + c2*B + c3*T*B + c4*T^2*B
```

## Scheduler 建议

推荐第一版 scheduler 的 cost lookup 逻辑：

```text
primary:
  table lookup by (latent_tokens, effective_batch_size)

fallback:
  step_ms = c0 + c1*T + c2*B + c3*T*B + c4*T^2*B

debug/conservative:
  保留 shape_key 统计字段，方便后续验证极端 shape 或异常 tail latency。
```

不要在第一版 median cost model 中把 `aspect_ratio` 作为主特征。它可以作为 debug/fallback 特征保留，但本次实验没有显示稳定的泛化收益。

## 复现步骤

在 910B 上：

```bash
cd /home/lzg/vllm-omni
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /home/lzg/venvs/vllm-omni-matrix-021-asc72643e/bin/activate

RUN_ID=20260522_215548 \
OUTDIR=/home/lzg/vllm-omni/benchmark_outputs/latent_token_aspect_profile_20260522_215548 \
PORT=18130 \
MASTER_PORT=26130 \
DEVICES=0,1 \
TP_SIZE=2 \
MAX_NUM_SEQS=4 \
STEPS=50 \
BATCH_SIZES=1,2,3,4 \
SQUARE_REPEATS=3 \
RECT_REPEATS=1 \
REQUEST_TIMEOUT_S=1200 \
HEALTH_TIMEOUT_S=1800 \
bash ./run_910b_latent_token_aspect_profile.sh
```

若已有 raw，只重新聚合：

```bash
python benchmarks/diffusion/step_cost_profile.py aggregate \
  --input benchmark_outputs/latent_token_aspect_profile_20260522_215548/step_cost_raw.jsonl \
  --output-dir benchmark_outputs/latent_token_aspect_profile_20260522_215548 \
  --manifest benchmark_outputs/latent_token_aspect_profile_20260522_215548/profile_run_manifest.jsonl

python benchmarks/diffusion/step_cost_profile.py analyze-latent-aspect \
  --input benchmark_outputs/latent_token_aspect_profile_20260522_215548/step_cost_raw.jsonl \
  --output-dir benchmark_outputs/latent_token_aspect_profile_20260522_215548 \
  --manifest benchmark_outputs/latent_token_aspect_profile_20260522_215548/profile_run_manifest.jsonl
```
