# High-load Scale4/6 Caveat Repeat Adaptive

This GitHub-safe bundle validates the two high-load scale4/6 caveats from `e2e_slo_frontier_high_load_scale46_adaptive_full` by adding repeat 1 and repeat 2 for `large-heavy` and `rectangular-mix`. It intentionally omits runtime logs, raw JSON/JSONL, traces, raw profiles, benchmark result JSON files, and large artifacts; omitted raw/runtime files are listed in `omitted_artifacts_manifest.csv`.

No scheduler code was changed for this run. The policies remain no step-level preemption and use the lookup cost model at `benchmarks/diffusion/profile_results/latent_token_aspect/step_cost_model.json`.

## Experiment Matrix

- New repeats: `1, 2`
- Prior repeat used for combined means: `0` from `benchmark_outputs/e2e_slo_frontier_high_load_scale46_adaptive_full`
- Workloads: `large-heavy, rectangular-mix`
- Interarrival seconds: `2.5`
- SLO scales: `4, 6`
- Policies: `slo_no_preemption_lookup, slo_no_preemption_guarded, slo_no_preemption_adaptive_guarded`
- `current` is intentionally not run in this focused repeat package.

## Conclusions

The repeat1/2 evidence does not reproduce the `large-heavy` scale4 adaptive miss disadvantage. Repeat1/2 mean miss is adaptive 44.2% vs lookup 47.5%; combined repeat0-2 is adaptive 44.4% vs lookup 45.6%. Adaptive also keeps better combined goodput, throughput, and P95 than lookup, but still has much higher bucket>=3 pressure (37.6% vs lookup 20.8%).

The `rectangular-mix` scale6 guarded全面领先 caveat is not stable. Repeat1/2 mean miss is adaptive 17.5% vs guarded 25.0%; combined repeat0-2 is adaptive 19.4% vs guarded 22.8%. Adaptive also has better combined goodput and P95, while guarded keeps a very small throughput edge (0.2225 vs adaptive 0.2197).

The two supporting points remain favorable to adaptive. `large-heavy` scale6 combined repeat0-2 has adaptive best miss/goodput/throughput/P95 (29.4% / 0.1331 / 0.1886 / 194.5s), and `rectangular-mix` scale4 also has adaptive best miss/goodput/throughput/P95 (36.7% / 0.1264 / 0.1991 / 165.4s). Bucket pressure remains the tradeoff to watch.

Each table cell below is `miss / goodput / throughput / P95 / bucket>=3`.

## Repeat1/2 Means

| Workload | Scale | Lookup | Guarded | Adaptive |
|---|---:|---:|---:|---:|
| large-heavy | 4 | 47.5% / 0.0822 / 0.1476 / 244.1s / 23.3% | 47.5% / 0.0882 / 0.1682 / 218.6s / 29.8% | 44.2% / 0.0896 / 0.1609 / 231.4s / 35.9% |
| large-heavy | 6 | 32.5% / 0.1102 / 0.1589 / 247.1s / 18.3% | 36.7% / 0.0924 / 0.1461 / 249.0s / 25.1% | 32.5% / 0.1279 / 0.1893 / 182.7s / 25.1% |
| rectangular-mix | 4 | 42.5% / 0.1071 / 0.1864 / 170.0s / 6.4% | 38.3% / 0.1208 / 0.1941 / 169.0s / 4.1% | 35.8% / 0.1379 / 0.2149 / 138.9s / 5.9% |
| rectangular-mix | 6 | 28.3% / 0.1552 / 0.2134 / 167.7s / 7.8% | 25.0% / 0.1662 / 0.2215 / 156.0s / 5.7% | 17.5% / 0.1807 / 0.2189 / 128.0s / 5.2% |

## Combined Repeat0-2 Means

| Workload | Scale | Lookup | Guarded | Adaptive |
|---|---:|---:|---:|---:|
| large-heavy | 4 | 45.6% / 0.0820 / 0.1420 / 255.3s / 20.8% | 49.4% / 0.0799 / 0.1534 / 233.9s / 27.9% | 44.4% / 0.0875 / 0.1577 / 231.9s / 37.6% |
| large-heavy | 6 | 32.2% / 0.1167 / 0.1692 / 229.8s / 20.6% | 37.2% / 0.0965 / 0.1539 / 232.1s / 24.5% | 29.4% / 0.1331 / 0.1886 / 194.5s / 29.5% |
| rectangular-mix | 4 | 43.9% / 0.1079 / 0.1928 / 173.3s / 6.7% | 39.4% / 0.1150 / 0.1885 / 175.8s / 3.6% | 36.7% / 0.1264 / 0.1991 / 165.4s / 7.6% |
| rectangular-mix | 6 | 26.1% / 0.1556 / 0.2089 / 162.4s / 8.7% | 22.8% / 0.1718 / 0.2225 / 149.7s / 7.2% | 19.4% / 0.1770 / 0.2197 / 138.1s / 6.3% |

## Shape-level Miss Highlights

| Question | Shape | Baseline | Adaptive |
|---|---|---:|---:|
| large-heavy scale4, combined adaptive vs lookup | 1024x1024 | 48/108 lookup | 43/108 adaptive |
| large-heavy scale4, combined adaptive vs lookup | 512x512 | 22/36 lookup | 25/36 adaptive |
| large-heavy scale4, combined adaptive vs lookup | 768x768 | 12/36 lookup | 12/36 adaptive |
| rectangular scale6, combined adaptive vs guarded | 512x768 | 10/27 guarded | 5/27 adaptive |
| rectangular scale6, combined adaptive vs guarded | 512x512 | 9/24 guarded | 5/24 adaptive |
| rectangular scale6, combined adaptive vs guarded | 1024x1024 | 4/24 guarded | 6/24 adaptive |

Full shape-level data is in `miss_by_shape_repeat1_2.csv` and `miss_by_shape_combined_repeat0_2.csv`.

## Files

- `repeat1_2_per_run.csv`: all 24 newly-run repeat1/2 rows.
- `mean_by_policy_repeat1_2.csv`: repeat1/2 policy means.
- `combined_repeat0_2_per_run.csv`: prior repeat0 plus new repeat1/2 rows for the focused matrix.
- `combined_repeat0_2_mean_by_policy.csv`: combined repeat0-2 policy means.
- `adaptive_pairwise_deltas.csv`: adaptive vs lookup/guarded deltas for repeat1/2 and combined repeat0-2.
- `frontier_table.csv`, `frontier_by_threshold.csv`, `frontier_report.md`: summaries generated from the new repeat1/2 raw output.
- `miss_by_shape*.csv`: shape-level miss summaries.
- `bucket_size_distribution.csv`: denoise bucket-size counts from the new repeat1/2 step profiles.
- `stagepool_replica_distribution.csv`: selected replica counts from the new repeat1/2 StagePool profiles.
- `figures/*.svg`: compact combined repeat0-2 visualizations.
- `omitted_artifacts_manifest.csv`: raw/runtime artifacts intentionally left out of this package.
