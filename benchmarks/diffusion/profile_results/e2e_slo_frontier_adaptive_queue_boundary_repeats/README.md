# E2E SLO Frontier Adaptive Queue Boundary Repeats

This is a lightweight, GitHub-safe result bundle built from the completed raw OUTDIR. Runtime logs, raw profiles, per-run `benchmark_result.json`, trace artifacts, PID files, runner scripts, and other raw artifacts are intentionally omitted and enumerated in `omitted_artifacts_manifest.csv`.

## Experiment Matrix

- Workloads: `large-heavy, bursty`
- Interarrival seconds: `2.5`
- SLO scales: `4.0`
- Repeats: `1, 2`
- Policies: `slo_no_preemption_guarded, slo_no_preemption_adaptive_guarded`

## Per-Run Results

| workload | policy | repeat | miss | goodput | throughput | P95 | failed | mean bucket |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bursty | adaptive | 1 | 31.7% | 0.1422 | 0.2081 | 148.80s | 0 | 1.771 |
| bursty | adaptive | 2 | 31.7% | 0.1398 | 0.2046 | 152.56s | 0 | 1.710 |
| bursty | guarded | 1 | 31.7% | 0.1221 | 0.1786 | 211.79s | 0 | 1.691 |
| bursty | guarded | 2 | 45.0% | 0.1146 | 0.2084 | 145.85s | 0 | 1.679 |
| large-heavy | adaptive | 1 | 43.3% | 0.1220 | 0.2152 | 192.26s | 0 | 2.114 |
| large-heavy | adaptive | 2 | 43.3% | 0.1036 | 0.1829 | 196.94s | 0 | 1.914 |
| large-heavy | guarded | 1 | 50.0% | 0.0709 | 0.1418 | 269.50s | 0 | 1.766 |
| large-heavy | guarded | 2 | 50.0% | 0.0922 | 0.1845 | 177.86s | 0 | 1.873 |

## Repeat Means

| workload | policy | miss | goodput | throughput | P95 | mean bucket |
|---|---|---:|---:|---:|---:|---:|
| bursty | adaptive | 31.7% | 0.1410 | 0.2063 | 150.68s | 1.741 |
| bursty | guarded | 38.3% | 0.1183 | 0.1935 | 178.82s | 1.685 |
| large-heavy | adaptive | 43.3% | 0.1128 | 0.1991 | 194.60s | 2.014 |
| large-heavy | guarded | 50.0% | 0.0816 | 0.1631 | 223.68s | 1.819 |

## Guarded vs Adaptive Repeat Means

| workload | adaptive miss delta | goodput change | throughput change | P95 change | bucket delta |
|---|---:|---:|---:|---:|---:|
| bursty | -6.7 pp | +19.1% | +6.6% | -15.7% | +0.056 |
| large-heavy | -6.7 pp | +38.3% | +22.0% | -13.0% | +0.195 |

## Broad Pilot Repeat0 Stability Check

| workload | policy | miss broad -> repeats | goodput change vs broad | throughput change vs broad | P95 change vs broad | bucket broad -> repeats |
|---|---|---:|---:|---:|---:|---:|
| bursty | adaptive | 23.3% -> 31.7% | -14.8% | -4.4% | -2.2% | 1.673 -> 1.741 |
| large-heavy | adaptive | 41.7% -> 43.3% | +20.0% | +23.6% | -18.6% | 1.902 -> 2.014 |

## Shape-Level Repeat Means

| workload | policy | shape | misses/requests | miss rate | P95 |
|---|---|---|---:|---:|---:|
| bursty | adaptive | 1024x1024 | 12/24 | 50.0% | 166.72s |
| bursty | adaptive | 512x512 | 18/48 | 37.5% | 148.65s |
| bursty | adaptive | 768x768 | 8/48 | 16.7% | 101.51s |
| bursty | guarded | 1024x1024 | 14/24 | 58.3% | 217.02s |
| bursty | guarded | 512x512 | 14/48 | 29.2% | 195.57s |
| bursty | guarded | 768x768 | 18/48 | 37.5% | 146.83s |
| large-heavy | adaptive | 1024x1024 | 23/72 | 31.9% | 196.05s |
| large-heavy | adaptive | 512x512 | 18/24 | 75.0% | 202.01s |
| large-heavy | adaptive | 768x768 | 11/24 | 45.8% | 141.99s |
| large-heavy | guarded | 1024x1024 | 27/72 | 37.5% | 255.97s |
| large-heavy | guarded | 512x512 | 21/24 | 87.5% | 214.40s |
| large-heavy | guarded | 768x768 | 12/24 | 50.0% | 232.99s |

## Bucket Summary

| workload | policy | mean bucket | bucket=3 share | counts across repeats |
|---|---|---:|---:|---|
| bursty | adaptive | 1.741 | 733/3448 (21.3%) | {'1': 1629, '2': 1086, '3': 733} |
| bursty | guarded | 1.685 | 813/3561 (22.8%) | {'1': 1935, '2': 813, '3': 813} |
| large-heavy | adaptive | 2.014 | 1093/2986 (36.6%) | {'1': 1065, '2': 828, '3': 1093} |
| large-heavy | guarded | 1.819 | 990/3301 (30.0%) | {'1': 1592, '2': 719, '3': 990} |

## Files

- `per_run_results.csv`: one row per completed run.
- `repeat_means_by_workload_policy.csv`: repeat-mean metrics by workload and policy.
- `guarded_vs_adaptive.csv`: adaptive deltas against guarded at repeat-mean level.
- `broad_pilot_repeat0_compare.csv`: boundary repeats compared with broad pilot repeat 0.
- `miss_by_shape.csv`: shape-level per-run miss/latency.
- `shape_repeat_means.csv`: shape-level repeat means.
- `bucket_size_distribution.csv`: per-run denoise bucket counts.
- `stagepool_replica_distribution.csv`: per-run StagePool selected replica counts.
- `figures/*.svg`: compact plots for miss, goodput, throughput, P95 latency, and mean bucket size.
- `omitted_artifacts_manifest.csv`: every raw OUTDIR file not copied into this package.
