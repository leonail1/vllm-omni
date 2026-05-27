# E2E SLO Throughput Frontier

This directory contains a lightweight, GitHub-safe result bundle for the no-preemption DiT SLO scheduler throughput-frontier experiment. Runtime logs, raw step profiles, stagepool profiles, trace files, trace sidecars, per-run `benchmark_result.json` files, and other runtime artifacts are intentionally omitted and listed in `omitted_artifacts_manifest.csv`.

The experiment varies offered load (`interarrival_s`) and SLO scale to see whether the scheduler can keep miss rate low while batching more same-shape work.

## Experiment Matrix

- Workloads: `large-heavy, rectangular-mix, bursty`
- Interarrival seconds: `2.5`
- SLO scales: `4, 6`
- Repeats: `0`
- Policies: `slo_no_preemption_guarded, slo_no_preemption_adaptive_guarded`

## Guarded vs Adaptive

Use `guarded_vs_adaptive.csv` for direct deltas against `slo_no_preemption_guarded`. The generated `current_vs_slo_no_preemption_adaptive_guarded.csv` is schema-only in this pilot because `current` was not included in the matrix.

## Files

- `frontier_summary.json`: structured summary from the raw output directory.
- `frontier_table.csv`: aggregate policy x workload x load x scale table.
- `frontier_by_threshold.csv`: best throughput under configured miss-rate thresholds.
- `guarded_vs_adaptive.csv`: direct deltas for adaptive versus guarded.
- `current_vs_slo_no_preemption_adaptive_guarded.csv`: generated schema-only comparison against `current` because `current` is not part of this pilot.
- `miss_by_shape.csv`: shape-level miss rate and latency.
- `bucket_size_distribution.csv`: denoise bucket size counts from step profiles.
- `stagepool_replica_distribution.csv`: selected replica counts from StagePool profiles.
- `figures/*.svg`: compact visualizations for miss rate, goodput, throughput, P95 latency, and mean bucket size.
