# E2E SLO Frontier Boundary Scale3 Repeats Adaptive

GitHub-safe package for focused repeat1/2 validation of the scale `3.0` boundary cases for `slo_no_preemption_adaptive_guarded`. Runtime logs, raw step and StagePool profiles, traces, trace manifests, JSON summaries, and per-run `benchmark_result.json` files are omitted and accounted for in `omitted_artifacts_manifest.csv`.

## Matrix

- Workloads: `large-heavy, rectangular-mix`
- Interarrival seconds: `2.5`
- SLO scales: `3.0`
- Policies: `current, slo_no_preemption_lookup, slo_no_preemption_guarded, slo_no_preemption_adaptive_guarded`
- Candidate policy: `slo_no_preemption_adaptive_guarded`
- Focused repeats: `1,2`
- Combined comparison: repeat `0` from `e2e_slo_frontier_boundary_high_load_adaptive` plus focused repeats `1,2`
- Raw focused output: `/home/lzg/vllm-omni/benchmark_outputs/e2e_slo_frontier_boundary_scale3_repeats_adaptive`
- Prior repeat0 source: `/home/lzg/vllm-omni/benchmark_outputs/e2e_slo_frontier_boundary_high_load_adaptive`

## Command

```bash
cd /home/lzg/vllm-omni
OUTDIR=/home/lzg/vllm-omni/benchmark_outputs/e2e_slo_frontier_boundary_scale3_repeats_adaptive PROFILE_MODE=e2e_slo_frontier_boundary_scale3_repeats_adaptive WORKLOADS=large-heavy,rectangular-mix INTERARRIVALS=2.5 SCALES=3.0 REPEATS=1,2 POLICIES=current,slo_no_preemption_lookup,slo_no_preemption_guarded,slo_no_preemption_adaptive_guarded CANDIDATE_POLICY=slo_no_preemption_adaptive_guarded PORT_BASE=21320 MASTER_PORT_BASE=29320 bash benchmarks/diffusion/run_910b_e2e_slo_frontier.sh
```

Runner defaults used: `MODEL=Qwen/Qwen-Image`, `VENV=/home/lzg/venvs/vllm-omni-matrix-021-asc72643e`, `DEVICES=0,1,2,3,4,5,6,7`, `REPLICAS=4`, `TP_SIZE=2`, `MAX_NUM_SEQS=4`, `NUM_REQUESTS=60`, `COST_MODEL=benchmarks/diffusion/profile_results/latent_token_aspect/step_cost_model.json`, no step-level preemption for SLO policies, lookup cost model.

## Conclusion

Large-heavy scale `3.0`: repeat1/2 adaptive mean miss is 50.0%, P95 201.90s, mean bucket 1.943, bucket>=3 33.0%. Combined repeat0-2 adaptive mean is 47.2% miss, P95 223.60s, mean bucket 2.007, bucket>=3 35.0%. This keeps the repeat0 pattern stable: adaptive has lower miss than guarded by 6.7 pp combined, but it packs harder and has higher P95 than guarded by 2.4%.
Large-heavy adaptive shape misses, repeat1/2: 1024x1024 31/72 (43.1%), 512x512 19/24 (79.2%), 768x768 10/24 (41.7%). Combined repeat0-2: 1024x1024 44/108 (40.7%), 512x512 31/36 (86.1%), 768x768 10/36 (27.8%).
Rectangular-mix scale `3.0`: repeat1/2 adaptive mean miss is 50.0%, P95 121.58s, throughput 0.2120. Combined adaptive miss is 51.1%, versus guarded 50.6% and lookup 48.9%; adaptive still trades worse miss than guarded by 0.6 pp, while improving combined P95 versus guarded by 34.9%.
Practical read: both scale3 caveats are stable enough to treat as real boundary behavior rather than repeat0 noise. Large-heavy remains an over-pack/shape-imbalance case; rectangular-mix remains a throughput/tail win with SLO under-protection relative to guarded.

## New Rows

| Workload | Policy | Repeat | Miss | Goodput | Throughput | P95 | Mean Bucket | Bucket>=3 | Failed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| large-heavy | current | 1 | 73.3% | 0.0428 | 0.1604 | 220.06s | 1.437 | 9.6% | 0 |
| large-heavy | current | 2 | 75.0% | 0.0378 | 0.1513 | 229.04s | 1.361 | 9.1% | 0 |
| large-heavy | lookup | 1 | 60.0% | 0.0718 | 0.1796 | 224.66s | 1.931 | 25.2% | 0 |
| large-heavy | lookup | 2 | 53.3% | 0.0661 | 0.1346 | 277.99s | 1.926 | 24.7% | 3 |
| large-heavy | guarded | 1 | 55.0% | 0.0779 | 0.1731 | 211.29s | 1.688 | 20.4% | 0 |
| large-heavy | guarded | 2 | 55.0% | 0.0771 | 0.1713 | 204.04s | 1.804 | 25.3% | 0 |
| large-heavy | adaptive | 1 | 48.3% | 0.1008 | 0.1952 | 192.50s | 1.999 | 37.8% | 0 |
| large-heavy | adaptive | 2 | 51.7% | 0.0809 | 0.1674 | 211.30s | 1.888 | 28.6% | 0 |
| rectangular-mix | current | 1 | 75.0% | 0.0482 | 0.1927 | 145.49s | 1.053 | 0.0% | 0 |
| rectangular-mix | current | 2 | 68.3% | 0.0453 | 0.1431 | 240.74s | 1.034 | 1.7% | 0 |
| rectangular-mix | lookup | 1 | 50.0% | 0.0965 | 0.1930 | 150.91s | 1.390 | 4.0% | 0 |
| rectangular-mix | lookup | 2 | 41.7% | 0.1024 | 0.1756 | 211.43s | 1.339 | 5.9% | 0 |
| rectangular-mix | guarded | 1 | 50.0% | 0.1004 | 0.2008 | 155.09s | 1.325 | 4.9% | 0 |
| rectangular-mix | guarded | 2 | 53.3% | 0.0912 | 0.1955 | 142.86s | 1.451 | 8.1% | 0 |
| rectangular-mix | adaptive | 1 | 46.7% | 0.1033 | 0.1936 | 132.66s | 1.337 | 1.5% | 0 |
| rectangular-mix | adaptive | 2 | 53.3% | 0.1075 | 0.2303 | 110.51s | 1.351 | 4.5% | 0 |

## Repeat1/2 Means

| Workload | Policy | Repeats | Miss | Goodput | Throughput | P95 | Mean Bucket | Bucket>=3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| large-heavy | current | 1;2 | 74.2% | 0.0403 | 0.1558 | 224.55s | 1.399 | 9.3% |
| large-heavy | lookup | 1;2 | 56.7% | 0.0690 | 0.1571 | 251.33s | 1.928 | 25.0% |
| large-heavy | guarded | 1;2 | 55.0% | 0.0775 | 0.1722 | 207.67s | 1.746 | 22.8% |
| large-heavy | adaptive | 1;2 | 50.0% | 0.0909 | 0.1813 | 201.90s | 1.943 | 33.0% |
| rectangular-mix | current | 1;2 | 71.7% | 0.0468 | 0.1679 | 193.11s | 1.044 | 0.9% |
| rectangular-mix | lookup | 1;2 | 45.8% | 0.0995 | 0.1843 | 181.17s | 1.364 | 5.0% |
| rectangular-mix | guarded | 1;2 | 51.7% | 0.0958 | 0.1981 | 148.98s | 1.388 | 6.4% |
| rectangular-mix | adaptive | 1;2 | 50.0% | 0.1054 | 0.2120 | 121.58s | 1.344 | 3.0% |

## Combined Repeat0-2 Means

| Workload | Policy | Repeats | Miss | Goodput | Throughput | P95 | Mean Bucket | Bucket>=3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| large-heavy | current | 0;1;2 | 77.2% | 0.0359 | 0.1583 | 217.33s | 1.441 | 11.2% |
| large-heavy | lookup | 0;1;2 | 57.8% | 0.0714 | 0.1683 | 242.32s | 1.910 | 23.7% |
| large-heavy | guarded | 0;1;2 | 53.9% | 0.0757 | 0.1645 | 218.28s | 1.744 | 24.7% |
| large-heavy | adaptive | 0;1;2 | 47.2% | 0.0866 | 0.1648 | 223.60s | 2.007 | 35.0% |
| rectangular-mix | current | 0;1;2 | 70.0% | 0.0502 | 0.1690 | 194.61s | 1.041 | 0.6% |
| rectangular-mix | lookup | 0;1;2 | 48.9% | 0.0966 | 0.1901 | 175.43s | 1.327 | 4.1% |
| rectangular-mix | guarded | 0;1;2 | 50.6% | 0.0888 | 0.1794 | 188.27s | 1.404 | 7.5% |
| rectangular-mix | adaptive | 0;1;2 | 51.1% | 0.1063 | 0.2186 | 122.54s | 1.347 | 3.2% |

## Files

- `new_runs_by_repeat.csv`: focused repeat1/2 per-run rows.
- `new_mean_by_policy.csv` and `frontier_table.csv`: focused repeat1/2 means by workload/policy.
- `repeat0_prior_boundary_rows.csv`: prior repeat0 rows used for combined views.
- `combined_runs_repeat0_1_2.csv`: combined per-run rows.
- `combined_repeat0_1_2_by_policy.csv` and `results_by_workload_load_policy_scale.csv`: combined repeat0-2 means.
- `adaptive_vs_baselines_new.csv` and `adaptive_vs_baselines_combined.csv`: adaptive deltas against current, lookup, and guarded.
- `miss_by_shape.csv`: shape-level miss and latency rows for repeat0/1/2.
- `bucket_size_distribution.csv` and `bucket_summary_by_policy.csv`: denoise bucket distributions and derived bucket>=3 summaries.
- `stagepool_replica_distribution.csv`: selected StagePool replica counts from summary-derived profile data.
- `figures/*.svg`: miss, goodput, throughput, P95, and mean-bucket policy charts.
- `omitted_artifacts_manifest.csv`: omitted raw/runtime artifact accounting for new and repeat0 raw output directories.
