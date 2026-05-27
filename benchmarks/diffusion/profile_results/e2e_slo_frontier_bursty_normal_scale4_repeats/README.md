# E2E SLO Frontier Bursty Normal Scale4 Repeats

GitHub-safe package for focused repeats on the `bursty`, interarrival `4.25`, scale `4.0` boundary. Runtime logs, raw step/stagepool files, traces, trace sidecars, per-run benchmark result JSON files, and profile/raw artifacts are omitted and listed in `omitted_artifacts_manifest.csv`.

## Matrix

- Workload: `bursty`
- Interarrival: `4.25`
- SLO scale: `4.0`
- Policies: `current, slo_no_preemption_lookup, slo_no_preemption_guarded, slo_no_preemption_adaptive_guarded`
- Candidate policy: `slo_no_preemption_adaptive_guarded`
- Focused repeats: `1,2`
- Combined comparison: repeat `0` from `e2e_slo_frontier_broad_normal_load_adaptive` plus focused repeats `1,2`
- Raw focused output: `/home/lzg/vllm-omni/benchmark_outputs/e2e_slo_frontier_bursty_normal_scale4_repeats_adaptive`
- Prior repeat0 source: `/home/lzg/vllm-omni/benchmark_outputs/e2e_slo_frontier_broad_normal_load_adaptive`

## Command

```bash
cd /home/lzg/vllm-omni
OUTDIR=/home/lzg/vllm-omni/benchmark_outputs/e2e_slo_frontier_bursty_normal_scale4_repeats_adaptive PROFILE_MODE=e2e_slo_frontier_bursty_normal_scale4_repeats_adaptive WORKLOADS=bursty INTERARRIVALS=4.25 SCALES=4.0 REPEATS=1,2 POLICIES=current,slo_no_preemption_lookup,slo_no_preemption_guarded,slo_no_preemption_adaptive_guarded CANDIDATE_POLICY=slo_no_preemption_adaptive_guarded PORT_BASE=20120 MASTER_PORT_BASE=28120 bash benchmarks/diffusion/run_910b_e2e_slo_frontier.sh
```

Runner defaults used: `MODEL=Qwen/Qwen-Image`, `VENV=/home/lzg/venvs/vllm-omni-matrix-021-asc72643e`, `DEVICES=0,1,2,3,4,5,6,7`, `REPLICAS=4`, `TP_SIZE=2`, `MAX_NUM_SEQS=4`, `NUM_REQUESTS=60`, `COST_MODEL=benchmarks/diffusion/profile_results/latent_token_aspect/step_cost_model.json`, no step-level preemption for SLO policies, lookup cost model.

## Conclusion

Repeat1/2 do not reproduce the repeat0 adaptive regression against current/lookup; the caveat looks like variance more than a stable boundary.

Repeat0 caveat from the prior broad package: adaptive miss 11.7%, P95 87.27s; current miss 10.0%, P95 82.91s; lookup miss 10.0%, P95 80.68s.

Focused repeat1/2 adaptive mean: miss 7.5%, goodput 0.1920, throughput 0.2076, P95 85.34s. Combined repeat0-2 adaptive mean: miss 8.9%, goodput 0.1859, throughput 0.2040, P95 85.98s.

## New Rows

| Policy | Repeat | Miss | Goodput | Throughput | P95 | Failed |
|---|---:|---:|---:|---:|---:|---:|
| current | 1 | 18.3% | 0.1463 | 0.1791 | 104.92s | 0 |
| current | 2 | 31.7% | 0.1201 | 0.1757 | 136.72s | 0 |
| lookup | 1 | 13.3% | 0.1841 | 0.2124 | 89.25s | 0 |
| lookup | 2 | 16.7% | 0.1722 | 0.2067 | 118.66s | 0 |
| guarded | 1 | 15.0% | 0.1506 | 0.1772 | 147.88s | 0 |
| guarded | 2 | 15.0% | 0.1760 | 0.2071 | 95.94s | 0 |
| adaptive | 1 | 6.7% | 0.1884 | 0.2018 | 83.67s | 0 |
| adaptive | 2 | 8.3% | 0.1956 | 0.2134 | 87.01s | 0 |

## Repeat1/2 Means

| Policy | Repeats | Miss | Goodput | Throughput | P95 |
|---|---:|---:|---:|---:|---:|
| current | 2 | 25.0% | 0.1332 | 0.1774 | 120.82s |
| lookup | 2 | 15.0% | 0.1782 | 0.2095 | 103.96s |
| guarded | 2 | 15.0% | 0.1633 | 0.1922 | 121.91s |
| adaptive | 2 | 7.5% | 0.1920 | 0.2076 | 85.34s |

## Combined Repeat0-2 Means

| Policy | Repeats | Miss | Goodput | Throughput | P95 |
|---|---:|---:|---:|---:|---:|
| current | 3 | 20.0% | 0.1438 | 0.1794 | 108.18s |
| lookup | 3 | 13.3% | 0.1781 | 0.2056 | 96.20s |
| guarded | 3 | 17.8% | 0.1546 | 0.1877 | 125.17s |
| adaptive | 3 | 8.9% | 0.1859 | 0.2040 | 85.98s |

## Files

- `new_runs_by_repeat.csv`: focused repeat1/2 per-run rows.
- `new_mean_by_policy.csv` and `frontier_table.csv`: focused repeat1/2 means by policy.
- `repeat0_prior_broad_rows.csv`: prior broad package repeat0 rows used for the combined view.
- `combined_runs_repeat0_1_2.csv`: combined per-run rows.
- `combined_repeat0_1_2_by_policy.csv` and `results_by_workload_load_policy_scale.csv`: combined means by policy.
- `adaptive_vs_baselines_combined.csv`: adaptive deltas against current, lookup, and guarded on repeat0-2.
- `miss_by_shape.csv`: shape-level misses for repeat0/1/2.
- `bucket_size_distribution.csv`: bucket-size distribution for repeat0/1/2.
- `stagepool_replica_distribution.csv`: selected replica counts from StagePool summaries.
- `figures/*.svg`: miss, goodput, throughput, and P95 policy charts.
- `omitted_artifacts_manifest.csv`: omitted runtime/raw artifact accounting.
