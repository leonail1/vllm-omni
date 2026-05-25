# E2E SLO Throughput Frontier

This directory contains a lightweight, GitHub-safe result bundle for the no-preemption DiT SLO scheduler throughput-frontier experiment. Runtime logs, raw step profiles, stagepool profiles, and trace text files are intentionally omitted and listed in `omitted_artifacts_manifest.csv`.

The experiment varies offered load (`interarrival_s`) and SLO scale to see whether the scheduler can keep miss rate low while batching more same-shape work.

## Experiment Matrix

- Workloads: `current-mix, shape-grouped-current-mix`
- Interarrival seconds: `4.25, 2.5`
- SLO scales: `4, 6`
- Repeats: `0`
- Policies: `current, slo_no_preemption_lookup`

## Current vs Candidate

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|
| current-mix | 2.50s | 4 | 8.33 pp | -5.42% | -17.87% | 56.03% | +0.402 |
| current-mix | 2.50s | 6 | -8.33 pp | -13.16% | -3.92% | 27.61% | +0.514 |
| current-mix | 4.25s | 4 | 25.00 pp | 61.66% | 18.36% | -19.24% | -0.033 |
| current-mix | 4.25s | 6 | 13.33 pp | 53.17% | 32.75% | -55.35% | -0.092 |
| shape-grouped-current-mix | 2.50s | 4 | -3.33 pp | -16.62% | -12.34% | 48.48% | +0.164 |
| shape-grouped-current-mix | 2.50s | 6 | -5.00 pp | -6.03% | -0.60% | 1.69% | +0.204 |
| shape-grouped-current-mix | 4.25s | 4 | 0.00 pp | 3.96% | 3.96% | -51.15% | -0.448 |
| shape-grouped-current-mix | 4.25s | 6 | 1.67 pp | 19.90% | 17.90% | -73.09% | -0.456 |

## Files

- `frontier_summary.json`: structured summary from the raw output directory.
- `frontier_table.csv`: aggregate policy x workload x load x scale table.
- `frontier_by_threshold.csv`: best throughput under configured miss-rate thresholds.
- `current_vs_slo_no_preemption_lookup.csv`: relative deltas against `current`.
- `miss_by_shape.csv`: shape-level miss rate and latency.
- `bucket_size_distribution.csv`: denoise bucket size counts from step profiles.
- `stagepool_replica_distribution.csv`: selected replica counts from StagePool profiles.
- `figures/*.svg`: compact visualizations for miss rate, goodput, throughput, P95 latency, and mean bucket size.
- `benchmark_results/`: small `benchmark_result.json` files only.
- `trace_manifests/`: trace manifests without the full trace text.
