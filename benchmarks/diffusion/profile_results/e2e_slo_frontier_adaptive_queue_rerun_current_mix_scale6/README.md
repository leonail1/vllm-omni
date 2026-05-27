# E2E SLO Throughput Frontier

This directory contains a lightweight, GitHub-safe result bundle for the no-preemption DiT SLO scheduler throughput-frontier experiment. Runtime logs, raw step profiles, stagepool profiles, and trace text files are intentionally omitted and listed in `omitted_artifacts_manifest.csv`.

The experiment varies offered load (`interarrival_s`) and SLO scale to see whether the scheduler can keep miss rate low while batching more same-shape work.

## Experiment Matrix

- Workloads: `current-mix`
- Interarrival seconds: `4.25, 2.5`
- SLO scales: `6`
- Repeats: `1, 2`
- Policies: `slo_no_preemption_guarded, slo_no_preemption_adaptive_guarded`

## Current vs slo_no_preemption_adaptive_guarded

This focused rerun does not repeat `current`, so `current_vs_slo_no_preemption_adaptive_guarded.csv` is a header-only schema placeholder. Compare guarded vs adaptive with `results_by_workload_load_policy_scale.csv`, `miss_by_shape.csv`, and `bucket_size_distribution.csv`.

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|

## Files

- `frontier_summary.json`: structured summary from the raw output directory.
- `frontier_table.csv`: aggregate policy x workload x load x scale table.
- `frontier_by_threshold.csv`: best throughput under configured miss-rate thresholds.
- `current_vs_slo_no_preemption_adaptive_guarded.csv`: header-only schema placeholder because this focused rerun does not include `current`.
- `miss_by_shape.csv`: shape-level miss rate and latency.
- `bucket_size_distribution.csv`: denoise bucket size counts from step profiles.
- `stagepool_replica_distribution.csv`: selected replica counts from StagePool profiles.
- `figures/*.svg`: compact visualizations for miss rate, goodput, throughput, P95 latency, and mean bucket size.
