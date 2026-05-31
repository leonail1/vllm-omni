# DiT SLO PARD actual-drop 小矩阵

本目录是 GitHub-safe 结果包，用于评估 PARD-style 主动丢弃是否值得纳入 DiT 主干 SLO 调度。raw step profile、StagePool profile、trace 文本和日志均未纳入，遗漏清单见 `omitted_artifacts_manifest.csv`。

## 实验设置

- 机器：远端 910B。
- 模型：`Qwen/Qwen-Image`。
- replica：4 个 DiT replica；每个 replica 使用 TP=2，因此总共使用 8 张卡。
- 请求数：每个点 60 个请求，repeat=0。
- 到达间隔：`ia=2.5`，即相邻请求按 trace 设定每 2.5 秒到达一次。
- workload：`current-mix` 为不同 shape 混合交错到达；`shape-grouped-current-mix` 为同一批 shape 分组到达。
- scale：SLO deadline = lookup reference cost x scale；本包包含 `scale=4` 和 `scale=6`。

## 对比策略

- `current`：vLLM-Omni 当前 DiT 调度对照。
- `slo_no_preemption_guarded`：baseline guarded 策略；不切走已驻留/运行请求，只在当前 bucket 内安全接纳新请求。
- `slo_no_preemption_adaptive_guarded`：在 baseline guarded 上加入 shape/deadline-aware 队列保护。
- `slo_no_preemption_pard_dry_run`：只记录 PARD oracle，不实际丢弃。
- `slo_no_preemption_pard_admission_drop`：只丢弃尚未驻留的等待请求，条件是 lookup cost 估计下即使单独执行也已无法满足 deadline。
- `slo_no_preemption_pard_step_drop`：允许在 denoise step 边界丢弃已驻留请求；不会在 kernel 执行中途打断。

## 主要结论

actual-drop 路径验证了主动丢弃链路可用：failed 请求不再卡到 300 秒超时，最大 failed latency 为 153.333 秒。但当前丢弃条件过激，整体不应替代 `slo_no_preemption_adaptive_guarded`：它显著降低 P95，却把大量请求转成 failed/miss，导致 miss rate 在多数点变差。

| workload | scale | policy | miss | goodput | throughput | P95(s) | failed |
|---|---:|---|---:|---:|---:|---:|---:|
| `current-mix` | 4.0 | `current` | 0.383 | 0.1211 | 0.1964 | 160.67 |  |
| `current-mix` | 4.0 | `slo_no_preemption_guarded` | 0.350 | 0.1349 | 0.2075 | 171.84 |  |
| `current-mix` | 4.0 | `slo_no_preemption_adaptive_guarded` | 0.333 | 0.1692 | 0.2537 | 113.27 |  |
| `current-mix` | 4.0 | `slo_no_preemption_pard_admission_drop` | 0.367 | 0.1962 | 0.1962 | 56.15 | 22 |
| `current-mix` | 4.0 | `slo_no_preemption_pard_step_drop` | 0.483 | 0.1493 | 0.1493 | 33.91 | 29 |
| `current-mix` | 6.0 | `current` | 0.167 | 0.1398 | 0.1677 | 177.60 |  |
| `current-mix` | 6.0 | `slo_no_preemption_guarded` | 0.283 | 0.1358 | 0.1895 | 150.22 |  |
| `current-mix` | 6.0 | `slo_no_preemption_adaptive_guarded` | 0.117 | 0.1901 | 0.2152 | 135.97 |  |
| `current-mix` | 6.0 | `slo_no_preemption_pard_admission_drop` | 0.633 | 0.1265 | 0.1265 | 44.47 | 38 |
| `current-mix` | 6.0 | `slo_no_preemption_pard_step_drop` | 0.400 | 0.1685 | 0.1685 | 60.38 | 24 |
| `shape-grouped-current-mix` | 4.0 | `current` | 0.350 | 0.1410 | 0.2169 | 115.12 |  |
| `shape-grouped-current-mix` | 4.0 | `slo_no_preemption_guarded` | 0.267 | 0.1533 | 0.2090 | 125.22 |  |
| `shape-grouped-current-mix` | 4.0 | `slo_no_preemption_adaptive_guarded` | 0.183 | 0.2079 | 0.2546 | 98.69 |  |
| `shape-grouped-current-mix` | 4.0 | `slo_no_preemption_pard_admission_drop` | 0.383 | 0.1703 | 0.1703 | 71.80 | 23 |
| `shape-grouped-current-mix` | 4.0 | `slo_no_preemption_pard_step_drop` | 0.400 | 0.1632 | 0.1632 | 73.40 | 24 |
| `shape-grouped-current-mix` | 6.0 | `current` | 0.017 | 0.2315 | 0.2354 | 97.22 |  |
| `shape-grouped-current-mix` | 6.0 | `slo_no_preemption_guarded` | 0.033 | 0.2446 | 0.2531 | 109.33 |  |
| `shape-grouped-current-mix` | 6.0 | `slo_no_preemption_adaptive_guarded` | 0.017 | 0.2423 | 0.2464 | 96.23 |  |
| `shape-grouped-current-mix` | 6.0 | `slo_no_preemption_pard_admission_drop` | 0.200 | 0.2390 | 0.2390 | 88.26 | 12 |
| `shape-grouped-current-mix` | 6.0 | `slo_no_preemption_pard_step_drop` | 0.283 | 0.2327 | 0.2327 | 78.08 | 17 |

## 文件

- `results_by_workload_load_policy_scale.csv`：策略 x workload x scale 的聚合指标。
- `pard_drop_vs_baselines.csv`：actual-drop 相对 current、baseline guarded、adaptive guarded、dry-run 的变化。
- `pard_drop_request_summary.csv`：actual-drop 的 completed/failed/SLO 命中数、failed latency 和 failed shape 分布。
- `pard_drop_debug_summary.csv`：从 step profile 派生的 drop 事件摘要。
- `miss_by_shape.csv`：shape-level miss 和 latency。
- `bucket_size_distribution.csv`、`stagepool_replica_distribution.csv`：bucket 与 replica 分布。
- `figures/*.svg`：miss、goodput、throughput、P95 和 mean bucket size 图。
- `benchmark_results/`：仅保留小型 `benchmark_result.json`。
- `trace_manifests/`：仅保留 trace manifest，不包含完整 trace 文本。
