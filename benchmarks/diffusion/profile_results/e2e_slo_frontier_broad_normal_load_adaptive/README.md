# E2E SLO Broad Normal-Load Adaptive Queue Guard

This directory contains a lightweight, GitHub-safe result bundle for the broad normal-load validation of `slo_no_preemption_adaptive_guarded`. Runtime logs, raw step/stagepool profiles, traces, trace sidecars, PID files, per-run `benchmark_result.json` files, and other raw runtime artifacts are intentionally omitted and listed in `omitted_artifacts_manifest.csv`.

## Experiment Matrix

- Workloads: `large-heavy, rectangular-mix, bursty`
- Interarrival: `4.25`
- SLO scales: `4.0, 6.0`
- Policies: `current, slo_no_preemption_lookup, slo_no_preemption_guarded, slo_no_preemption_adaptive_guarded`
- Repeat: `0`
- Candidate policy: `slo_no_preemption_adaptive_guarded`

## Rows

| Workload | Policy | Scale | Miss | Goodput | Throughput | P95 | Failed |
|---|---|---:|---:|---:|---:|---:|---:|
| large-heavy | current | 4 | 38.3% | 0.1094 | 0.1773 | 117.94s | 0 |
| large-heavy | slo_no_preemption_lookup | 4 | 20.0% | 0.1375 | 0.1718 | 117.57s | 0 |
| large-heavy | slo_no_preemption_guarded | 4 | 18.3% | 0.1417 | 0.1735 | 116.57s | 0 |
| large-heavy | slo_no_preemption_adaptive_guarded | 4 | 18.3% | 0.1585 | 0.1941 | 96.55s | 0 |
| large-heavy | current | 6 | 25.0% | 0.1040 | 0.1386 | 181.78s | 0 |
| large-heavy | slo_no_preemption_lookup | 6 | 5.0% | 0.1771 | 0.1864 | 97.94s | 0 |
| large-heavy | slo_no_preemption_guarded | 6 | 10.0% | 0.1523 | 0.1693 | 117.22s | 0 |
| large-heavy | slo_no_preemption_adaptive_guarded | 6 | 5.0% | 0.1841 | 0.1938 | 99.75s | 0 |
| rectangular-mix | current | 4 | 30.0% | 0.1115 | 0.1593 | 105.96s | 0 |
| rectangular-mix | slo_no_preemption_lookup | 4 | 0.0% | 0.2017 | 0.2017 | 53.94s | 0 |
| rectangular-mix | slo_no_preemption_guarded | 4 | 0.0% | 0.1974 | 0.1974 | 55.09s | 0 |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 4 | 0.0% | 0.1993 | 0.1993 | 54.02s | 0 |
| rectangular-mix | current | 6 | 10.0% | 0.1572 | 0.1746 | 104.17s | 0 |
| rectangular-mix | slo_no_preemption_lookup | 6 | 0.0% | 0.2049 | 0.2049 | 44.28s | 0 |
| rectangular-mix | slo_no_preemption_guarded | 6 | 0.0% | 0.1913 | 0.1913 | 71.99s | 0 |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 6 | 0.0% | 0.1999 | 0.1999 | 56.71s | 0 |
| bursty | current | 4 | 10.0% | 0.1652 | 0.1835 | 82.91s | 0 |
| bursty | slo_no_preemption_lookup | 4 | 10.0% | 0.1779 | 0.1977 | 80.68s | 0 |
| bursty | slo_no_preemption_guarded | 4 | 23.3% | 0.1371 | 0.1788 | 131.69s | 0 |
| bursty | slo_no_preemption_adaptive_guarded | 4 | 11.7% | 0.1738 | 0.1967 | 87.27s | 0 |
| bursty | current | 6 | 1.7% | 0.2019 | 0.2053 | 89.07s | 0 |
| bursty | slo_no_preemption_lookup | 6 | 3.3% | 0.1857 | 0.1921 | 91.63s | 0 |
| bursty | slo_no_preemption_guarded | 6 | 10.0% | 0.1674 | 0.1860 | 132.33s | 0 |
| bursty | slo_no_preemption_adaptive_guarded | 6 | 0.0% | 0.2003 | 0.2003 | 85.78s | 0 |

## Files

- `frontier_summary.json`: structured summary from the raw output directory.
- `frontier_table.csv` and `results_by_workload_load_policy_scale.csv`: aggregate policy x workload x load x scale tables.
- `broad_normal_current_lookup_guarded_adaptive.csv`: flattened 24-row matrix.
- `adaptive_vs_broad_normal_baselines.csv`: adaptive deltas against current, lookup, and guarded.
- `current_vs_slo_no_preemption_adaptive_guarded.csv`: adaptive deltas against current from the frontier summarizer.
- `miss_by_shape.csv`: shape-level request/miss counts.
- `bucket_size_distribution.csv`: denoise bucket size counts from step profiles.
- `stagepool_replica_distribution.csv`: selected replica counts from StagePool profiles.
- `figures/*.svg`: compact visualizations for miss rate, goodput, throughput, P95 latency, and mean bucket size.
- `omitted_artifacts_manifest.csv`: raw OUTDIR files intentionally not packaged.
