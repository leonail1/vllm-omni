# E2E SLO Workload Sweep

This directory contains a lightweight, GitHub-safe result bundle. Runtime logs, raw step profiles, stagepool profiles, and trace text files are intentionally omitted and listed in `omitted_artifacts_manifest.csv`.

## Experiment Matrix

- Workloads: `current-mix, large-heavy, rectangular-mix, bursty`
- SLO scales: `2.5, 3, 3.5, 4`
- Repeats: `0`
- Policies: `current, slo_no_preemption_lookup`

## Current vs Candidate

| Workload | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change |
|---|---:|---:|---:|---:|---:|
| bursty | 2.5 | 22.50 pp | 126.24% | 33.68% | -13.95% |
| bursty | 3 | -8.75 pp | -11.55% | 1.08% | 16.34% |
| bursty | 3.5 | -16.25 pp | -34.53% | -19.05% | 118.99% |
| bursty | 4 | 15.00 pp | 38.93% | 12.88% | -5.38% |
| current-mix | 2.5 | 20.00 pp | 82.33% | 35.28% | -41.37% |
| current-mix | 3 | 35.00 pp | 107.29% | 29.90% | -53.86% |
| current-mix | 3.5 | 28.75 pp | 105.71% | 46.57% | -69.98% |
| current-mix | 4 | 12.50 pp | 33.09% | 16.02% | -35.51% |
| large-heavy | 2.5 | 40.00 pp | 126.73% | -7.63% | 19.91% |
| large-heavy | 3 | 53.75 pp | 611.53% | 54.60% | -47.29% |
| large-heavy | 3.5 | 30.00 pp | 67.14% | 3.47% | 42.66% |
| large-heavy | 4 | 27.50 pp | 94.42% | 26.52% | -32.34% |
| rectangular-mix | 2.5 | 36.25 pp | 90.82% | 6.98% | -40.08% |
| rectangular-mix | 3 | 27.50 pp | 86.02% | 23.06% | -14.36% |
| rectangular-mix | 3.5 | 33.75 pp | 107.83% | 28.11% | -53.71% |
| rectangular-mix | 4 | 16.25 pp | 23.11% | 1.77% | -30.46% |

## Files

- `ablation_summary.json`: structured summary from the raw output directory.
- `ablation_table.csv`: aggregate policy x workload x scale table.
- `results_by_workload_policy_scale.csv`: flattened aggregate metrics.
- `current_vs_slo_no_preemption_lookup.csv`: relative deltas against `current`.
- `miss_by_shape.csv`: shape-level miss rate and latency.
- `bucket_size_distribution.csv`: denoise bucket size counts from step profiles.
- `stagepool_replica_distribution.csv`: selected replica counts from StagePool profiles.
- `figures/*.svg`: compact visualizations for miss rate, goodput, throughput, P95 latency, and mean bucket size.
- `benchmark_results/`: small `benchmark_result.json` files only.
- `trace_manifests/`: trace manifests without the full trace text.
