# PR4024 Dynamic Batching Baseline Smoke

这个结果包用于记录 PR4024 接到当前 `feat/dit-slo-cost-oracle` 基线后的最小诊断结论。它不是完整性能矩阵。

## 结论

- `smoke_without_cli_eager`：`current` 和 `pr4024_dynamic` 都能完成 12 个请求；但 `pr4024_dynamic` 的 step profile 里最大 batch size 仍为 1，说明动态 token batch 没有真正启用。
- `forced_cli_eager_default_backend`：补上 `--enforce-eager` 后，动态 key 生效并在 dummy run 阶段触发 mixed-shape dynamic batch；910B 默认 attention 后端是 `SDPA`，服务启动失败，报错 `Qwen-Image dynamic step batching requires FlashAttention backend`。
- `forced_cli_eager_flash_backend`：强制 `DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN` 后，后端解析为 `FLASH_ATTN`，但 NPU 实现需要 MindIE-SD，当前 910B 环境未安装，dummy run 仍失败。
- `current` 在 smoke 中观察到的最大 batch size 为 2，这是原有同 key/同 shape batch，不是 PR4024 的异构 shape 动态 token batch。
- 因此当前机器暂时不能得到有效的 PR4024 dynamic baseline；后续需要先补齐 MindIE-SD/Ascend FlashAttention 依赖，或实现 PR4024 的 SDPA/NPU varlen fallback。

## 实验设置

- 模型：`Qwen/Qwen-Image`
- DiT 主干 smoke：4 replica，`DEVICES=0,1,2,3`、`REPLICAS=4`、`TP_SIZE=1`、`MAX_NUM_SEQS=4`、12 个请求、`current-mix`、到达间隔 `ia=2.5`、`scale=4.0`。
- 强制排队诊断：单 replica、`DEVICES=0`、`MAX_NUM_SEQS=4`、`current-mix`、到达间隔 `ia=0.1`、`scale=4.0`；默认后端诊断 8 个请求，强制 FlashAttention 诊断 4 个请求。
- 指标：`miss_rate` 是 SLO 未达成比例；`goodput_rps` 是满足 SLO 的成功请求吞吐；`throughput_rps` 是总成功请求吞吐；`latency_p95_s` 是端到端 P95 延迟。

## 文件说明

- `summary.csv`：可运行 smoke 与两个强制 eager 诊断的核心结果。
- `dynamic_batch_profile.csv`：step profile 中观察到的 batch size 分布摘要。
- `frontier_table.csv` / `frontier_by_threshold.csv` / `current_vs_pr4024_dynamic.csv`：benchmark 脚本生成的轻量 CSV。
- `figures/*.svg`：核心指标图。
- `omitted_artifacts_manifest.csv`：未纳入结果包的日志、JSONL、PID、raw profile 文件清单。
