# E2E SLO Throughput Frontier

This directory contains a lightweight, GitHub-safe result bundle for the no-preemption DiT SLO scheduler throughput-frontier experiment. Runtime logs, raw step profiles, stagepool profiles, and trace text files are intentionally omitted and listed in `omitted_artifacts_manifest.csv`.

The experiment varies offered load (`interarrival_s`) and SLO scale to see whether the scheduler can keep miss rate low while batching more same-shape work.

## Experiment Matrix

- Workloads: `current-mix, shape-grouped-current-mix`
- Interarrival seconds: `4.25, 2.5`
- SLO scales: `4, 6`
- Repeats: `0`
- Policies: `current, slo_no_preemption_lookup, slo_no_preemption_guarded`

## Current vs slo_no_preemption_guarded

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|
| current-mix | 2.50s | 4 | 11.67 pp | 27.63% | 5.29% | -5.61% | +0.332 |
| current-mix | 2.50s | 6 | -13.33 pp | -8.89% | 7.68% | -5.33% | +0.448 |
| current-mix | 4.25s | 4 | 30.00 pp | 70.48% | 18.47% | -41.05% | -0.096 |
| current-mix | 4.25s | 6 | 13.33 pp | 63.15% | 41.39% | -58.78% | -0.101 |
| shape-grouped-current-mix | 2.50s | 4 | 8.33 pp | 16.63% | 3.96% | 19.35% | +0.094 |
| shape-grouped-current-mix | 2.50s | 6 | 0.00 pp | 2.91% | 2.91% | -12.85% | -0.102 |
| shape-grouped-current-mix | 4.25s | 4 | 0.00 pp | 4.08% | 4.08% | -53.10% | -0.481 |
| shape-grouped-current-mix | 4.25s | 6 | 1.67 pp | 19.86% | 17.86% | -73.43% | -0.472 |

## Files

- `frontier_summary.json`: structured summary from the raw output directory.
- `frontier_table.csv`: aggregate policy x workload x load x scale table.
- `frontier_by_threshold.csv`: best throughput under configured miss-rate thresholds.
- `current_vs_slo_no_preemption_guarded.csv`: relative deltas against `current`.
- `miss_by_shape.csv`: shape-level miss rate and latency.
- `bucket_size_distribution.csv`: denoise bucket size counts from step profiles.
- `stagepool_replica_distribution.csv`: selected replica counts from StagePool profiles.
- `figures/*.svg`: compact visualizations for miss rate, goodput, throughput, P95 latency, and mean bucket size.
- `benchmark_results/`: small `benchmark_result.json` files only.
- `trace_manifests/`: trace manifests without the full trace text.
