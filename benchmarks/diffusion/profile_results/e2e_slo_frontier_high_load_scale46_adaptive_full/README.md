# High-load Scale4/6 Adaptive Full SLO Frontier

This directory contains the GitHub-safe result bundle for the high-load (`interarrival_s=2.5`) scale4/6 full four-policy completion package for the no-preemption DiT SLO scheduler frontier. It compares `current`, `slo_no_preemption_lookup`, `slo_no_preemption_guarded`, and `slo_no_preemption_adaptive_guarded` on `large-heavy`, `rectangular-mix`, and `bursty` with repeat 0. Runtime logs, raw step profiles, stagepool profiles, trace text, raw JSON summaries, and benchmark result JSON files are intentionally omitted and listed in `omitted_artifacts_manifest.csv`.

The experiment fixes high load at `interarrival_s=2.5` and sweeps SLO scale 4/6 to complete the baseline comparison for the adaptive guarded policy. No step-level preemption is enabled, and the lookup cost model is used.

## Experiment Matrix

- Workloads: `large-heavy, rectangular-mix, bursty`
- Interarrival seconds: `2.5`
- SLO scales: `4, 6`
- Repeats: `0`
- Policies: `current, slo_no_preemption_lookup, slo_no_preemption_guarded, slo_no_preemption_adaptive_guarded`


## Conclusion

Adaptive guarded is a useful high-load scale4/6 candidate, but this package is not a final proof. It has the best miss rate in 4 of 6 workload/scale points: `large-heavy` scale 6, `rectangular-mix` scale 4, and `bursty` scales 4 and 6. The strongest row is `bursty` scale 6, where adaptive has the best miss (8.3%), goodput (0.2389 rps), throughput (0.2607 rps), and P95 (118.8s).

The caveats are important. On `large-heavy` scale 4, adaptive misses 45.0%, which is worse than lookup at 41.7%, although adaptive has better goodput, throughput, and P95. On `rectangular-mix` scale 6, guarded is the best row: adaptive regresses miss (23.3% vs guarded 18.3%), goodput (0.1697 vs 0.1832), throughput (0.2213 vs 0.2243), and P95 (158.4s vs 137.1s). Large-heavy also still shows high packing pressure under adaptive, with bucket>=3 at 40.9% for scale 4 and 38.3% for scale 6.

Practical read: this completes the high-load scale4/6 baseline comparison and supports adaptive guarded as the better broad candidate than shape_guarded, especially for bursty and large-heavy scale 6. It still leaves two boundary caveats to carry forward rather than hiding them: large-heavy scale4 versus lookup, and rectangular scale6 versus guarded.

## Row Table

| Workload | Scale | Current | Lookup | Guarded | Adaptive |
|---|---:|---:|---:|---:|---:|
| large-heavy | 4 | 63.3% / 0.0494 / 0.1279 / 261.3s / 11.7% | 41.7% / 0.0818 / 0.1308 / 277.5s / 15.9% | 53.3% / 0.0631 / 0.1240 / 264.7s / 24.2% | 45.0% / 0.0832 / 0.1513 / 232.7s / 40.9% |
| large-heavy | 6 | 33.3% / 0.1198 / 0.1797 / 179.5s / 9.7% | 31.7% / 0.1298 / 0.1899 / 195.3s / 25.2% | 38.3% / 0.1046 / 0.1697 / 198.4s / 23.2% | 23.3% / 0.1435 / 0.1872 / 218.1s / 38.3% |
| rectangular-mix | 4 | 51.7% / 0.0784 / 0.1623 / 201.8s / 0.0% | 46.7% / 0.1096 / 0.2055 / 180.0s / 7.3% | 41.7% / 0.1035 / 0.1774 / 189.3s / 2.7% | 38.3% / 0.1033 / 0.1674 / 218.6s / 10.9% |
| rectangular-mix | 6 | 31.7% / 0.1221 / 0.1787 / 162.8s / 0.0% | 21.7% / 0.1565 / 0.1998 / 151.6s / 10.6% | 18.3% / 0.1832 / 0.2243 / 137.1s / 10.0% | 23.3% / 0.1697 / 0.2213 / 158.4s / 8.5% |
| bursty | 4 | 40.0% / 0.1125 / 0.1875 / 163.7s / 2.1% | 46.7% / 0.1155 / 0.2166 / 161.6s / 23.6% | 40.0% / 0.1278 / 0.2130 / 180.3s / 24.9% | 36.7% / 0.1234 / 0.1948 / 162.2s / 23.3% |
| bursty | 6 | 36.7% / 0.0940 / 0.1484 / 284.2s / 1.9% | 26.7% / 0.1392 / 0.1898 / 187.2s / 26.7% | 20.0% / 0.1695 / 0.2119 / 157.5s / 31.2% | 8.3% / 0.2389 / 0.2607 / 118.8s / 21.2% |

Each cell is `miss / goodput / throughput / P95 / bucket>=3`.

## Current vs slo_no_preemption_adaptive_guarded

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|
| bursty | 2.50s | 4 | 3.33 pp | 9.71% | 3.94% | -0.88% | +0.468 |
| bursty | 2.50s | 6 | 28.33 pp | 154.29% | 75.69% | -58.19% | +0.769 |
| large-heavy | 2.50s | 4 | 18.33 pp | 68.46% | 18.22% | -10.92% | +0.606 |
| large-heavy | 2.50s | 6 | 10.00 pp | 19.81% | 4.18% | 21.48% | +0.557 |
| rectangular-mix | 2.50s | 4 | 13.33 pp | 31.64% | 3.18% | 8.33% | +0.302 |
| rectangular-mix | 2.50s | 6 | 8.33 pp | 38.97% | 23.86% | -2.69% | +0.288 |

## Files

- `frontier_table.csv`: aggregate policy x workload x load x scale table.
- `frontier_by_threshold.csv`: best throughput under configured miss-rate thresholds.
- `current_vs_slo_no_preemption_adaptive_guarded.csv`: relative deltas against `current`.
- `miss_by_shape.csv`: shape-level miss rate and latency.
- `bucket_size_distribution.csv`: denoise bucket size counts from step profiles.
- `stagepool_replica_distribution.csv`: selected replica counts from StagePool profiles.
- `figures/*.svg`: compact visualizations for miss rate, goodput, throughput, P95 latency, and mean bucket size.
- `omitted_artifacts_manifest.csv`: raw/runtime artifacts intentionally left out of this no-JSON package.
