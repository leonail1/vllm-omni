# First-Stage Token SLO Smoke

This GitHub-safe result bundle verifies that the split first-stage DiT token SLO code can start vLLM-Omni on 910B and complete a minimal request set. It is a smoke test, not a throughput-frontier or performance-claim experiment.

Runtime logs, raw step profiles, stagepool profiles, and raw trace text files are intentionally omitted and listed in `omitted_artifacts_manifest.csv`.

## Smoke Matrix

- Workload: `current-mix`
- Interarrival seconds: `2.5`
- SLO scale: `4.0`
- Repeat: `0`
- Requests per policy: `2`
- Replicas / TP: `2` replicas, `TP1`
- Model/runtime: Qwen-Image int8, FLASH_ATTN / MindIE-SD on 910B
- Policies: `slo_no_preemption_token_objective`, `slo_token_step_preemptive`

## Result

Both first-stage token SLO policies completed the smoke successfully:

| Policy | Completed | Failed | SLO miss rate | P95 latency |
|---|---:|---:|---:|---:|
| `slo_no_preemption_token_objective` | 2 | 0 | 0.0% | 14.08s |
| `slo_token_step_preemptive` | 2 | 0 | 0.0% | 14.43s |

This package does not claim that encoder/decoder DAG runtime or PARD is implemented in the first-stage commit. Those are second-stage work.

## Files

- `frontier_summary.json`: structured summary generated from the raw output directory.
- `frontier_table.csv` and `results_by_workload_load_policy_scale.csv`: aggregate smoke metrics.
- `miss_by_shape.csv`: shape-level smoke outcomes.
- `bucket_size_distribution.csv`: denoise bucket size counts from safe summarized profiles.
- `stagepool_replica_distribution.csv`: selected replica counts from safe summarized StagePool profiles.
- `figures/*.svg`: compact smoke visualizations.
- `benchmark_results/`: small `benchmark_result.json` files only.
- `trace_manifests/`: trace manifests without raw trace text.
- `omitted_artifacts_manifest.csv`: raw artifacts intentionally omitted.
