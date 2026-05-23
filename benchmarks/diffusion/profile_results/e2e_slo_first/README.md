# Qwen-Image E2E SLO Scheduler Experiment

本目录保存第一轮真实 E2E 实验结果，用来比较当前 vLLM-Omni diffusion scheduler 与第一版 `full_slo` 调度策略。

## 配置

- 机器：910B，8 NPU
- 模型：`Qwen/Qwen-Image`
- 服务形态：`4 replicas x TP=2`，每个 replica 使用 2 张 NPU
- `max_num_seqs=4`
- `num_inference_steps=50`
- 请求数：每个 policy/scale 各 80 个请求
- 到达间隔：全局固定 `4.25s`
- shape 分布：`512x512 : 768x768 : 1024x1024 = 32 : 32 : 16`
- SLO：`slo_ms = reference_cost_ms * slo_scale`
- cost model：`benchmarks/diffusion/profile_results/latent_token_aspect/step_cost_model.json`

`current` 是当前 FIFO/homogeneous batching 路径；`full_slo` 启用：

```json
{
  "diffusion_scheduler_policy": "slo",
  "diffusion_slo_scheduler": {
    "step_cost_model_path": "benchmarks/diffusion/profile_results/latent_token_aspect/step_cost_model.json",
    "step_cost_metric": "p90_ms",
    "step_cost_formula": "qwen_image_910b_tp2_v1",
    "batch_growth_alpha": 0.6
  }
}
```

没有设置 `step_cost_safety_factor`。

## 结果

| SLO scale | Policy | Miss rate | Goodput req/s | Throughput req/s | P95 latency | Mean bucket |
|---:|---|---:|---:|---:|---:|---:|
| 2.5 | current | 37.50% | 0.1224 | 0.1959 | 74.96s | 1.230 |
| 2.5 | full_slo | 2.50% | 0.2094 | 0.2148 | 40.75s | 1.135 |
| 4.0 | current | 36.25% | 0.1032 | 0.1619 | 170.47s | 1.199 |
| 4.0 | full_slo | 0.00% | 0.2191 | 0.2191 | 38.18s | 1.122 |
| 5.0 | current | 8.75% | 0.1608 | 0.1762 | 111.18s | 1.220 |
| 5.0 | full_slo | 0.00% | 0.2156 | 0.2156 | 38.88s | 1.120 |

相对提升：

| SLO scale | Miss 降低 | Goodput 提升 | Throughput 提升 | P95 latency ratio |
|---:|---:|---:|---:|---:|
| 2.5 | 35.00 pp | 1.71x | 1.10x | 0.54x |
| 4.0 | 36.25 pp | 2.12x | 1.35x | 0.22x |
| 5.0 | 8.75 pp | 1.34x | 1.22x | 0.35x |

## 为什么 miss rate 和 goodput 改善

这轮结果不支持“full_slo 靠显著增大 bucket size 获胜”的解释。相反，`full_slo` 的平均 bucket size 比 `current` 略小：

| SLO scale | current mean bucket | full_slo mean bucket |
|---:|---:|---:|
| 2.5 | 1.230 | 1.135 |
| 4.0 | 1.199 | 1.122 |
| 5.0 | 1.220 | 1.120 |

更合理的初步解释是：

1. `full_slo` 避免了 FIFO 队头阻塞，让紧 deadline 的请求更早进入 denoise step。
2. StagePool 根据 deadline/cost/load 选择 replica，减少请求被送到长尾 replica 的概率。
3. Instance scheduler 在 step boundary 按 laxity 选择 bucket，降低单个 shape 或单个老请求拖住其他请求的风险。
4. 因为长尾明显缩短，benchmark 总完成时间下降，所以 goodput 和 throughput 都提高。

按 shape 的 miss 也支持这个结论：`full_slo` 只有 `scale=2.5` 的 `768x768` 有 2 个 miss，其余 full_slo shape/scale 均为 0 miss。

本轮 raw profile 是补充 queue/deadline 字段之前跑出的旧格式，因此当前数据能直接支持 bucket size、shape、replica、step time 的分析；queue pressure、deadline pressure、per-replica busy/idle time 需要用更新后的 profiler 在下一轮消融中重新采集。

## 文件说明

- `analysis.json`：按 scale 的 current/full_slo 对比和 delta。
- `analysis_table.csv`：主要指标表，适合导入表格。
- `ABLATION_PLAN.md`：下一轮消融实验设计，以及需要新增采集的 request / StagePool / instance / bucket 指标。
- `summary.json` / `summary.md`：由修复后的 summarizer 重新生成的可信汇总。
- `current/`、`full_slo/`：每个 policy 的原始结果。
- `*/scale_*/benchmark_result.json`：底层 benchmark 输出。
- `*/step_cost_raw*.jsonl`：所有 replica 的 step-level raw profile。
- `runner.log`、`*/server.log`、`*/scale_*/client.log`：运行日志。

注意：原始 `benchmark_result.json` 中的 `metrics.slo_scale` 是旧 benchmark 脚本默认值 `3.0`，不应用作真实 scale。真实 scale 由 trace 中逐请求 `slo_ms/reference_cost_ms` 和 raw profile 的 `profile_scale` 确认，分别为 `2.5/4.0/5.0`。脚本已在后续修复，会把真实 `--slo-scale` 写入 metrics。

## 复现

在 910B 的 `/home/lzg/vllm-omni`：

```bash
OUTDIR=/home/lzg/vllm-omni/benchmark_outputs/e2e_slo_first \
NUM_REQUESTS=80 \
GLOBAL_INTERARRIVAL_S=4.25 \
SCALES=2.5,4.0,5.0 \
REPLICAS=4 \
TP_SIZE=2 \
MAX_NUM_SEQS=4 \
bash benchmarks/diffusion/run_910b_e2e_slo_first.sh
```

重新汇总：

```bash
python benchmarks/diffusion/e2e_slo_first_experiment.py summarize \
  --output-dir benchmarks/diffusion/profile_results/e2e_slo_first \
  --scales 2.5,4.0,5.0
```

## 已知限制

- 每个组合只跑了一次，没有 repeat，不能排除运行时波动。
- current 的 `scale=4.0` p95 latency 明显异常高，后续应重复验证。
- 当前 trace 是固定间隔到达，不覆盖 bursty/Poisson 流量。
- 当前 raw profile 不包含 queue/deadline pressure 字段；代码已补采集字段，下一轮实验需要重新跑。
- 本轮只覆盖 Qwen-Image text-to-image，不覆盖 Wan2.2/video。
