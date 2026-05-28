# E2E SLO Frontier Shape Guard Scale3 Pilot

GitHub-safe result bundle for the shape-aware StagePool queue guard alias `slo_no_preemption_shape_guarded`. Runtime logs, raw step profiles, StagePool JSONL profiles, traces, trace sidecars, PID files, and per-run `benchmark_result.json` files are omitted and accounted for in `omitted_artifacts_manifest.csv`.

## Experiment Matrix

- Workloads: `large-heavy`, `rectangular-mix`
- Interarrival seconds: `2.5`
- SLO scales: `3.0`
- Policies: `current`, `slo_no_preemption_lookup`, `slo_no_preemption_guarded`, `slo_no_preemption_adaptive_guarded`, `slo_no_preemption_shape_guarded`
- Candidate policy: `slo_no_preemption_shape_guarded`
- Repeats: `0`
- Requests per run: `60`
- Replicas: `4`, tensor parallel size: `2`, devices: `0,1,2,3,4,5,6,7`
- Cost model: `benchmarks/diffusion/profile_results/latent_token_aspect/step_cost_model.json`
- Step preemption: disabled for SLO no-preemption policies
- Output directory: `/home/lzg/vllm-omni/benchmark_outputs/e2e_slo_frontier_shape_guard_scale3_pilot`

## Command

```bash
cd /home/lzg/vllm-omni
OUTDIR=/home/lzg/vllm-omni/benchmark_outputs/e2e_slo_frontier_shape_guard_scale3_pilot PROFILE_MODE=e2e_slo_frontier_shape_guard_scale3_pilot WORKLOADS=large-heavy,rectangular-mix INTERARRIVALS=2.5 SCALES=3.0 REPEATS=0 POLICIES=current,slo_no_preemption_lookup,slo_no_preemption_guarded,slo_no_preemption_adaptive_guarded,slo_no_preemption_shape_guarded CANDIDATE_POLICY=slo_no_preemption_shape_guarded PORT_BASE=22620 MASTER_PORT_BASE=30620 bash benchmarks/diffusion/run_910b_e2e_slo_frontier.sh
```

## Conclusion

Shape-aware guarded is mixed at the scale `3.0` boundary. On `rectangular-mix`, it is the best miss-rate row at 45.0%, slightly better than lookup (46.7%) and much better than adaptive (56.7%); it also has the best goodput (0.1159 rps). The cost is tail latency versus adaptive: P95 is 144.4s versus adaptive 122.0s.

On `large-heavy`, shape-aware guarded does not improve the stable-boundary miss issue versus adaptive. Miss is 48.3% versus adaptive 43.3%, although P95 is slightly lower (232.8s versus 237.7s) and bucket>=3 is lower (33.3% versus 38.8%). Shape-level results explain the tradeoff: 1024x1024 improves (8/36 (22.2%) versus adaptive 11/36 (30.6%)), 512x512 remains fully missed (12/12 (100.0%) for both), and 768x768 regresses (9/12 (75.0%) versus adaptive 3/12 (25.0%)).

Practical read: the alias is promising for the rectangular mix, but it is not a clean fix for `large-heavy` at scale `3.0`; it shifts misses across shapes while still leaving small-shape starvation visible.

## Row Table

| Workload | Policy | Miss | Goodput | Throughput | P95 | Mean bucket | Bucket>=3 | Failed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| large-heavy | current | 73.3% | 0.0401 | 0.1503 | 218.0s | 1.350 | 8.5% | 0 |
| large-heavy | slo_no_preemption_lookup | 58.3% | 0.0570 | 0.1300 | 274.1s | 1.806 | 18.6% | 3 |
| large-heavy | slo_no_preemption_guarded | 55.0% | 0.0831 | 0.1847 | 191.5s | 1.797 | 24.9% | 0 |
| large-heavy | slo_no_preemption_adaptive_guarded | 43.3% | 0.0895 | 0.1579 | 237.7s | 2.062 | 38.8% | 0 |
| large-heavy | slo_no_preemption_shape_guarded | 48.3% | 0.0888 | 0.1719 | 232.8s | 2.035 | 33.3% | 0 |
| rectangular-mix | current | 58.3% | 0.0597 | 0.1434 | 240.3s | 1.053 | 0.0% | 0 |
| rectangular-mix | slo_no_preemption_lookup | 46.7% | 0.0993 | 0.1862 | 195.3s | 1.424 | 11.0% | 0 |
| rectangular-mix | slo_no_preemption_guarded | 55.0% | 0.1016 | 0.2258 | 158.4s | 1.261 | 6.2% | 0 |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 56.7% | 0.0961 | 0.2218 | 122.0s | 1.364 | 1.2% | 0 |
| rectangular-mix | slo_no_preemption_shape_guarded | 45.0% | 0.1159 | 0.2108 | 144.4s | 1.412 | 5.6% | 0 |

## Shape And Bucket Observations

- Large-heavy shape-guarded vs adaptive: 1024x1024 miss improves to 8/36 (22.2%) from 11/36 (30.6%); 512x512 remains 12/12 (100.0%); 768x768 worsens to 9/12 (75.0%) from 3/12 (25.0%).
- Large-heavy bucket>=3 drops from adaptive 38.8% to shape-guarded 33.3%, but shape-guarded still packs harder than guarded (24.9%) and lookup (18.6%).
- Rectangular-mix shape-guarded has bucket>=3 5.6%; this is higher than adaptive 1.2%, lower than lookup 11.0%, and close to guarded 6.2%.

## Caveats

- Single repeat (`0`) pilot; scheduler noise is not averaged.
- Offered load is intentionally high (`interarrival_s=2.5`), so miss rates are boundary signals rather than production SLO attainment.
- P5 time-to-deadline remains negative in every row.
- Raw artifacts are intentionally excluded; derived shape, bucket, and replica summaries are included.

## Files

- `shape_guard_scale3_matrix.csv`: compact 10-row matrix.
- `results_by_workload_load_policy_scale.csv` and `frontier_table.csv`: aggregate summaries from the frontier summarizer.
- `shape_guarded_vs_baselines.csv`: shape-guarded deltas against current, lookup, guarded, and adaptive.
- `shape_guarded_vs_adaptive_guarded.csv`: focused shape-guarded vs adaptive delta table.
- `current_vs_slo_no_preemption_shape_guarded.csv`: summarizer current-vs-candidate deltas.
- `miss_by_shape.csv`: shape-level miss and latency summary.
- `bucket_size_distribution.csv` and `bucket_summary_by_policy.csv`: denoise bucket distributions and derived bucket>=3 shares.
- `stagepool_replica_distribution.csv`: selected StagePool replica distribution.
- `figures/*.svg`: miss, goodput, throughput, P95, and mean-bucket figures.
- `omitted_artifacts_manifest.csv`: omitted raw/log/jsonl/trace/profile/benchmark_result artifacts from the raw OUTDIR.
