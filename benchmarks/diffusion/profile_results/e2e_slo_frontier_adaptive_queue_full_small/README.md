# E2E SLO Frontier Adaptive Queue Full Small

This directory contains a lightweight, GitHub-safe result bundle for the adaptive guarded no-preemption DiT SLO scheduler throughput-frontier experiment. Runtime logs, raw step profiles, stagepool profiles, PID files, trace files, and raw-run metadata are intentionally omitted and listed in `omitted_artifacts_manifest.csv`.

The experiment varies offered load (`interarrival_s`) and SLO scale to see whether the scheduler can keep miss rate low while batching more same-shape work.

## Experiment Matrix

- Workloads: `current-mix, shape-grouped-current-mix`
- Interarrival seconds: `4.25, 2.5`
- SLO scales: `4, 6`
- Repeats: `0`
- Policies: `slo_no_preemption_adaptive_guarded`

## Baseline Comparison

This package contains the adaptive policy rows for the full-small follow-up matrix. Baseline rows (`current`, `slo_no_preemption_lookup`, and `slo_no_preemption_guarded`) are not duplicated here; compare against `../e2e_slo_frontier_guarded/results_by_workload_load_policy_scale.csv` when evaluating deltas.

## Files

- `frontier_summary.json`: structured summary from the raw output directory.
- `frontier_table.csv`: aggregate policy x workload x load x scale table.
- `frontier_by_threshold.csv`: best throughput under configured miss-rate thresholds.
- `current_vs_slo_no_preemption_adaptive_guarded.csv`: header-only comparison schema for this package because baseline rows are not duplicated here.
- `miss_by_shape.csv`: shape-level miss rate and latency.
- `bucket_size_distribution.csv`: denoise bucket size counts from step profiles.
- `stagepool_replica_distribution.csv`: selected replica counts from StagePool profiles.
- `figures/*.svg`: compact visualizations for miss rate, goodput, throughput, P95 latency, and mean bucket size.
