# SLO Scheduler Ablation Plan

目标：解释 `full_slo` 为什么能降低 SLO miss rate，并区分 StagePool、instance scheduler、cost model、preemption/bucket 策略各自的贡献。

## 固定实验条件

- 模型：`Qwen/Qwen-Image`
- 硬件：910B，`4 replicas x TP=2`
- `max_num_seqs=4`
- `num_inference_steps=50`
- trace：沿用本目录的 80 请求 mixed-shape trace
- 重点 scale：`2.5` 和 `4.0`
- 每组至少 3 repeats；先跑 `scale=2.5` 快速筛选，再补 `scale=4.0`

注意：`e2e_slo_first` 目录中的旧 raw profile 不包含 queue/deadline pressure 字段。下面这些指标需要用更新后的 profiler 重新跑消融实验采集。

## 消融矩阵

| 组别 | StagePool | Instance scheduler | Cost model | 目的 |
|---|---|---|---|---|
| A | current | FIFO | 无 | 基线 |
| B | SLO-aware | FIFO | lookup | 只看 replica 选择贡献 |
| C | current | SLO-aware | lookup | 只看 denoise step 调度贡献 |
| D | SLO-aware | SLO-aware | constant cost | 验证 deadline 调度是否依赖精确 cost |
| E | SLO-aware | SLO-aware | formula fallback | 验证 lookup table 的必要性 |
| F | SLO-aware | SLO-aware | lookup | full_slo |
| G | SLO-aware | SLO-aware no preemption | lookup | 验证 step-level preemption/切 bucket 的贡献 |
| H | SLO-aware | SLO-aware | lookup, alpha=0/0.3/0.6/1.0 | 验证 batch growth cost 的敏感性 |

## 需要新增或重点采集的指标

### Request 级

- `assigned_replica_id`
- `arrival_time_s`
- `deadline_time_s`
- `first_step_start_s`
- `completion_time_s`
- `queue_wait_ms = first_step_start_s - arrival_time_s`
- `execution_ms = completion_time_s - first_step_start_s`
- `slack_at_completion_ms = deadline_time_s - completion_time_s`
- `slo_achieved`
- `shape_key`
- `makespan_s`

用途：判断 miss 降低是来自排队等待降低，还是来自执行时间降低。

### StagePool 级

- 每次入队时各 replica 的：
  - `queue_length`
  - `busy_time_ms`
  - `idle_time_ms`
  - `completed_requests`
  - `completed_steps`
  - `safe_admit_capacity`
  - `predicted_laxity_ms`
  - `estimated_admit_delay_ms`
  - `estimated_existing_laxity_after_ms`
- 最终选择的 `replica_id`
- `stagepool_policy`

用途：解释请求为什么被送到某个 replica，以及 replica 选择是否减少长尾。

### Instance scheduler 级

- 每个 scheduler tick 的候选 bucket：
  - `shape_key`
  - `candidate_batch_size`
  - `num_waiting`
  - `num_running`
  - `min_laxity_ms`
  - `min_laxity_ratio`
  - `estimated_step_ms`
  - `incremental_step_ms_if_add_one`
- 最终选择的 bucket 和未选择 bucket 的 score
- 是否发生 step-level preemption：当前 running bucket 是否被跳过

用途：解释 SLO scheduler 是否真正优先了紧 deadline bucket，以及是否牺牲/减少 batching。

### Bucket 执行级

本轮已经在 raw profile 中记录：

- `batch_size`
- `effective_batch_size`
- `shape_key`
- `step_indices`
- `denoise_ms`
- `step_scheduler_ms`
- `post_decode_ms`
- `total_step_ms`
- `replica_id`
- `tp_size`
- `max_num_seqs`

本次新增后续实验可用字段：

- `num_waiting_reqs`
- `num_running_reqs`
- `remaining_steps`
- `arrival_time_s`
- `deadline_time_s`
- `reference_cost_ms`
- `slo_ms`
- `age_ms`
- `time_to_deadline_ms`

用途：把每个实际执行的 bucket 与队列压力、deadline 压力和 cost model 预测关联起来。

## 关键判断问题

1. `full_slo` 的 miss 降低是否主要来自 queue wait 降低？
2. StagePool 只改 replica 选择时，是否能显著降低 p95 latency？
3. Instance scheduler 只改 bucket 选择时，是否能减少 shape-level 长尾？
4. cost lookup 换成 constant/formula 后，miss rate 是否明显反弹？
5. `batch_growth_alpha` 增大是否会带来更大 bucket，但伤害紧 deadline 请求？
6. preemption 关闭后，是否重新出现 FIFO 队头阻塞？

## 输出表

每组实验输出：

- `benchmark_result.json`
- `step_cost_raw*.jsonl`
- `ablation_summary.json`
- `ablation_table.csv`
- `ablation_report.md`

报告至少包含：

- miss rate / goodput / throughput / p95 latency
- mean/p95 queue wait
- mean/p95 execution time
- mean/p95 slack at completion
- mean bucket size / bucket distribution
- per-shape miss rate
- per-replica load and miss distribution
- per-replica busy/idle time
- per-replica completed requests / completed steps
- actual vs estimated step cost error
