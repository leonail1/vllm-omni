# E2E SLO Broad Current/Lookup Baseline

This directory contains a lightweight, GitHub-safe result bundle for the broad workload baseline complement to `e2e_slo_frontier_adaptive_queue_broad_pilot`. Runtime logs, raw step profiles, StagePool profiles, trace files, trace sidecars, PID files, per-run `benchmark_result.json` files, and other raw runtime artifacts are intentionally omitted and listed in `omitted_artifacts_manifest.csv`.

## Experiment Matrix

- Workloads: `large-heavy, rectangular-mix, bursty`
- Interarrival: `2.5`
- SLO scales: `4.0, 6.0`
- Policies: `current, slo_no_preemption_lookup`
- Repeat: `0`
- Candidate policy: `slo_no_preemption_lookup`

## Baseline Rows

| Workload | Policy | Scale | Miss | Goodput | Throughput | P95 | Failed |
|---|---|---:|---:|---:|---:|---:|---:|
| bursty | current | 4 | 48.3% | 0.0942 | 0.1823 | 175.73s | 0 |
| bursty | current | 6 | 10.0% | 0.1738 | 0.1931 | 147.10s | 0 |
| bursty | slo_no_preemption_lookup | 4 | 26.7% | 0.1524 | 0.2078 | 141.66s | 0 |
| bursty | slo_no_preemption_lookup | 6 | 26.7% | 0.1387 | 0.1891 | 189.11s | 0 |
| large-heavy | current | 4 | 56.7% | 0.0653 | 0.1506 | 256.59s | 0 |
| large-heavy | current | 6 | 41.7% | 0.1056 | 0.1810 | 186.58s | 0 |
| large-heavy | slo_no_preemption_lookup | 4 | 46.7% | 0.1016 | 0.1905 | 235.61s | 0 |
| large-heavy | slo_no_preemption_lookup | 6 | 36.7% | 0.0877 | 0.1292 | 273.61s | 4 |
| rectangular-mix | current | 4 | 50.0% | 0.0865 | 0.1731 | 171.37s | 0 |
| rectangular-mix | current | 6 | 33.3% | 0.1190 | 0.1785 | 166.27s | 0 |
| rectangular-mix | slo_no_preemption_lookup | 4 | 41.7% | 0.1303 | 0.2233 | 128.05s | 0 |
| rectangular-mix | slo_no_preemption_lookup | 6 | 28.3% | 0.1166 | 0.1626 | 219.49s | 0 |

## Comparison Files

- `broad_current_lookup_guarded_adaptive.csv`: current/lookup rows from this package aligned with guarded/adaptive rows from `../e2e_slo_frontier_adaptive_queue_broad_pilot`.
- `adaptive_vs_broad_baselines.csv`: adaptive deltas against current, lookup, and guarded for each workload/scale.

## Files

- `frontier_summary.json`: structured summary from the raw output directory.
- `frontier_table.csv` and `results_by_workload_load_policy_scale.csv`: aggregate policy x workload x load x scale table.
- `baseline_rows.csv`: one row per completed baseline run.
- `current_vs_slo_no_preemption_lookup.csv`: lookup deltas against current.
- `miss_by_shape.csv`: shape-level request/miss counts.
- `bucket_size_distribution.csv`: denoise bucket size counts from step profiles.
- `stagepool_replica_distribution.csv`: selected replica counts from StagePool profiles.
- `figures/*.svg`: compact visualizations over current, lookup, guarded, and adaptive.
- `omitted_artifacts_manifest.csv`: raw OUTDIR files intentionally not packaged.
