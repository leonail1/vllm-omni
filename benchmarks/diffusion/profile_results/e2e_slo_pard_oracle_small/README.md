# E2E SLO Throughput Frontier

This directory contains a lightweight, GitHub-safe result bundle for the no-preemption DiT SLO scheduler throughput-frontier experiment. Runtime logs, raw step profiles, stagepool profiles, and trace text files are intentionally omitted and listed in `omitted_artifacts_manifest.csv`.

The experiment varies offered load (`interarrival_s`) and SLO scale to see whether the scheduler can keep miss rate low while batching more same-shape work.

`slo_no_preemption_pard_dry_run` is `slo_no_preemption_adaptive_guarded` plus a PARD-style dry-run oracle. It records requests that would be dropped by an admission-only or step-boundary policy, but it does not actually reject or abort requests in this experiment.

## Experiment Matrix

- Workloads: `current-mix, shape-grouped-current-mix`
- Interarrival seconds: `2.5`
- SLO scales: `4, 6`
- Repeats: `0`
- Policies: `current, slo_no_preemption_guarded, slo_no_preemption_adaptive_guarded, slo_no_preemption_pard_dry_run`

## Current vs slo_no_preemption_pard_dry_run

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|
| current-mix | 2.50s | 4 | 11.67 pp | 49.19% | 25.46% | -28.88% | +0.376 |
| current-mix | 2.50s | 6 | 5.00 pp | 49.39% | 40.94% | -16.12% | +0.394 |
| shape-grouped-current-mix | 2.50s | 4 | 20.00 pp | 49.88% | 14.62% | -18.61% | -0.086 |
| shape-grouped-current-mix | 2.50s | 6 | -5.00 pp | -2.63% | 2.59% | 7.66% | +0.004 |

## Baseline Guarded vs PARD Dry Run

| Workload | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change |
|---|---:|---:|---:|---:|---:|
| current-mix | 4 | 8.33 pp | 33.91% | 18.70% | -33.50% |
| current-mix | 6 | 16.67 pp | 53.72% | 24.72% | -0.82% |
| shape-grouped-current-mix | 4 | 11.67 pp | 37.91% | 18.98% | -25.18% |
| shape-grouped-current-mix | 6 | -3.33 pp | -7.85% | -4.55% | -4.27% |

## Files

- `frontier_summary.json`: structured summary from the raw output directory.
- `frontier_table.csv`: aggregate policy x workload x load x scale table.
- `frontier_by_threshold.csv`: best throughput under configured miss-rate thresholds.
- `current_vs_slo_no_preemption_pard_dry_run.csv`: relative deltas against `current`.
- `baseline_guarded_vs_slo_no_preemption_pard_dry_run.csv`: relative deltas against `slo_no_preemption_guarded`.
- `pard_oracle_summary.csv`: dry-run oracle counts per workload and SLO scale.
- `pard_oracle_by_shape.csv`: dry-run oracle counts grouped by shape.
- `pard_oracle_records_sample.csv`: bounded sample of oracle records whose dry-run decision was positive.
- `miss_by_shape.csv`: shape-level miss rate and latency.
- `bucket_size_distribution.csv`: denoise bucket size counts from step profiles.
- `stagepool_replica_distribution.csv`: selected replica counts from StagePool profiles.
- `figures/*.svg`: compact visualizations for miss rate, goodput, throughput, P95 latency, and mean bucket size.
- `benchmark_results/`: small `benchmark_result.json` files only.
- `trace_manifests/`: trace manifests without the full trace text.
