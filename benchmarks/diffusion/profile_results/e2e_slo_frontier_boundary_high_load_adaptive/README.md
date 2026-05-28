# E2E SLO Frontier Boundary High-Load Adaptive

This GitHub-safe bundle summarizes the high-load boundary pilot for `slo_no_preemption_adaptive_guarded`. Runtime logs, raw step profiles, StagePool profiles, traces, per-run `benchmark_result.json` files, and trace sidecars are omitted and accounted for in `omitted_artifacts_manifest.csv`.

## Experiment Matrix

- Workloads: `large-heavy, rectangular-mix, bursty`
- Interarrival seconds: `2.5`
- SLO scales: `3.0, 3.5`
- Policies: `current, slo_no_preemption_lookup, slo_no_preemption_guarded, slo_no_preemption_adaptive_guarded`
- Candidate policy: `slo_no_preemption_adaptive_guarded`
- Repeats: `0`
- Requests per run: `60`
- Replicas: `4`, tensor parallel size: `2`, devices: `0,1,2,3,4,5,6,7`
- Cost model: `benchmarks/diffusion/profile_results/latent_token_aspect/step_cost_model.json`
- Step preemption: disabled for SLO no-preemption policies

## Conclusion

Adaptive guarded is strongest at scale `3.5` and for bursty at both scales, but the scale `3.0` boundary is mixed rather than a clean pass. At scale `3.5`, adaptive is best or tied on miss for all workloads and usually improves P95 versus lookup and guarded. At scale `3.0`, adaptive improves miss and goodput versus current and lookup, but large-heavy shows over-packing: mean bucket `2.133`, bucket>=3 share `39.3%`, best miss `41.7%`, but worst P95 `267.0s` and very negative P5 time-to-deadline. Rectangular-mix at scale `3.0` is the other caveat: adaptive has much better P95 and throughput than guarded, but misses `53.3%` versus guarded `48.3%`.

Practical read: adaptive guarded behaves well enough at `3.5`; at `3.0` it needs more guard tuning before it can be called robust, especially for large-heavy shape mix and the 512x512/1024x1024 split.

## Row Table

| Workload | Policy | Scale | Miss | Goodput | Throughput | P95 | Mean bucket | Failed/missing |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| large-heavy | current | 3 | 83.3% | 0.0272 | 0.1633 | 202.9s | 1.526 | 0 |
| large-heavy | slo_no_preemption_lookup | 3 | 60.0% | 0.0763 | 0.1908 | 224.3s | 1.873 | 0 |
| large-heavy | slo_no_preemption_guarded | 3 | 51.7% | 0.0721 | 0.1491 | 239.5s | 1.741 | 0 |
| large-heavy | slo_no_preemption_adaptive_guarded | 3 | 41.7% | 0.0781 | 0.1317 | 267.0s | 2.133 | 0 |
| large-heavy | current | 3.5 | 76.7% | 0.0378 | 0.1619 | 197.2s | 1.479 | 0 |
| large-heavy | slo_no_preemption_lookup | 3.5 | 63.3% | 0.0628 | 0.1712 | 249.2s | 2.001 | 0 |
| large-heavy | slo_no_preemption_guarded | 3.5 | 46.7% | 0.0784 | 0.1470 | 256.2s | 1.717 | 0 |
| large-heavy | slo_no_preemption_adaptive_guarded | 3.5 | 45.0% | 0.1061 | 0.1929 | 198.5s | 2.009 | 0 |
| rectangular-mix | current | 3 | 66.7% | 0.0571 | 0.1713 | 197.6s | 1.034 | 0 |
| rectangular-mix | slo_no_preemption_lookup | 3 | 55.0% | 0.0908 | 0.2018 | 163.9s | 1.252 | 0 |
| rectangular-mix | slo_no_preemption_guarded | 3 | 48.3% | 0.0746 | 0.1421 | 266.9s | 1.434 | 0 |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 3 | 53.3% | 0.1082 | 0.2318 | 124.5s | 1.353 | 0 |
| rectangular-mix | current | 3.5 | 66.7% | 0.0619 | 0.1857 | 159.0s | 1.049 | 0 |
| rectangular-mix | slo_no_preemption_lookup | 3.5 | 46.7% | 0.0947 | 0.1776 | 210.4s | 1.413 | 0 |
| rectangular-mix | slo_no_preemption_guarded | 3.5 | 50.0% | 0.1055 | 0.2110 | 133.4s | 1.345 | 0 |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 3.5 | 46.7% | 0.1073 | 0.2013 | 121.5s | 1.321 | 0 |
| bursty | current | 3 | 68.3% | 0.0516 | 0.1630 | 201.4s | 1.358 | 0 |
| bursty | slo_no_preemption_lookup | 3 | 53.3% | 0.0944 | 0.2022 | 149.0s | 1.813 | 0 |
| bursty | slo_no_preemption_guarded | 3 | 48.3% | 0.1186 | 0.2295 | 150.4s | 1.717 | 0 |
| bursty | slo_no_preemption_adaptive_guarded | 3 | 38.3% | 0.1451 | 0.2353 | 132.6s | 1.739 | 0 |
| bursty | current | 3.5 | 48.3% | 0.0944 | 0.1827 | 157.4s | 1.485 | 0 |
| bursty | slo_no_preemption_lookup | 3.5 | 53.3% | 0.0990 | 0.2121 | 175.7s | 1.715 | 0 |
| bursty | slo_no_preemption_guarded | 3.5 | 50.0% | 0.1094 | 0.2187 | 192.6s | 1.609 | 0 |
| bursty | slo_no_preemption_adaptive_guarded | 3.5 | 38.3% | 0.1545 | 0.2505 | 117.4s | 1.583 | 0 |

## Adaptive vs Baselines

Positive miss reduction means adaptive missed less than the baseline. Negative P95 change means adaptive was faster at P95.

| Workload | Scale | Baseline | Miss reduction | Goodput change | Throughput change | P95 change | Bucket delta |
|---|---:|---|---:|---:|---:|---:|---:|
| large-heavy | 3 | current | +41.7 pp | +187.1% | -19.3% | +31.6% | +0.607 |
| large-heavy | 3 | slo_no_preemption_lookup | +18.3 pp | +2.4% | -31.0% | +19.0% | +0.260 |
| large-heavy | 3 | slo_no_preemption_guarded | +10.0 pp | +8.4% | -11.7% | +11.5% | +0.392 |
| large-heavy | 3.5 | current | +31.7 pp | +180.9% | +19.2% | +0.7% | +0.531 |
| large-heavy | 3.5 | slo_no_preemption_lookup | +18.3 pp | +69.0% | +12.7% | -20.3% | +0.008 |
| large-heavy | 3.5 | slo_no_preemption_guarded | +1.7 pp | +35.3% | +31.2% | -22.5% | +0.292 |
| rectangular-mix | 3 | current | +13.3 pp | +89.5% | +35.3% | -37.0% | +0.318 |
| rectangular-mix | 3 | slo_no_preemption_lookup | +1.7 pp | +19.1% | +14.9% | -24.1% | +0.101 |
| rectangular-mix | 3 | slo_no_preemption_guarded | -5.0 pp | +44.9% | +63.2% | -53.4% | -0.082 |
| rectangular-mix | 3.5 | current | +20.0 pp | +73.5% | +8.4% | -23.6% | +0.272 |
| rectangular-mix | 3.5 | slo_no_preemption_lookup | 0.0 pp | +13.3% | +13.3% | -42.3% | -0.092 |
| rectangular-mix | 3.5 | slo_no_preemption_guarded | +3.3 pp | +1.8% | -4.6% | -9.0% | -0.024 |
| bursty | 3 | current | +30.0 pp | +181.1% | +44.4% | -34.2% | +0.381 |
| bursty | 3 | slo_no_preemption_lookup | +15.0 pp | +53.7% | +16.3% | -11.0% | -0.074 |
| bursty | 3 | slo_no_preemption_guarded | +10.0 pp | +22.3% | +2.5% | -11.8% | +0.022 |
| bursty | 3.5 | current | +10.0 pp | +63.7% | +37.1% | -25.4% | +0.098 |
| bursty | 3.5 | slo_no_preemption_lookup | +15.0 pp | +56.1% | +18.1% | -33.2% | -0.132 |
| bursty | 3.5 | slo_no_preemption_guarded | +11.7 pp | +41.2% | +14.5% | -39.0% | -0.026 |

## Shape And Bucket Observations

- Large-heavy scale `3.0`: adaptive over-packs relative to guarded and lookup. It has the largest mean bucket (`2.133`) and bucket>=3 share (`39.3%`). Shape misses are polarized: `1024x1024` improves to `13/36`, `768x768` improves to `0/12`, but `512x512` is `12/12` misses and P95 is about `272s`.
- Large-heavy scale `3.5`: adaptive still packs heavily (`2.009` mean bucket, `35.8%` bucket>=3), but this scale gives enough slack that it is best on miss (`45.0%`) and throughput (`0.1929 rps`) with P95 near current and far below lookup/guarded.
- Rectangular-mix scale `3.0`: adaptive is less aggressive than guarded on bucket>=3 (`3.5%` vs `9.8%`) and has much lower P95 (`124.5s` vs `266.9s`), but misses more (`53.3%` vs `48.3%`). This looks like a throughput/tail improvement with some SLO under-protection, not a simple over-pack.
- Rectangular-mix scale `3.5`: adaptive is conservative on buckets (`1.321` mean, `3.4%` bucket>=3), ties lookup on miss (`46.7%`), beats lookup P95 strongly (`121.5s` vs `210.4s`), and slightly beats guarded miss/P95.
- Bursty: adaptive looks healthy at both scales. Bucket sizes are close to guarded or lower, miss is best (`38.3%` at both scales), and P95 is lowest among SLO no-preemption policies at scale `3.5`.

## Caveats

- This is a single repeat (`0`) pilot, so row-to-row scheduler noise is not averaged out.
- Offered load is intentionally tight (`interarrival_s=2.5`); all policies still miss heavily, so conclusions are about boundary behavior rather than production-ready SLO attainment.
- P5 time-to-deadline is negative for every row, meaning the workload is beyond the practical no-miss frontier at these scales.
- Raw profiles are omitted from the package by design; derived shape, bucket, and replica CSVs are included.

## Files

- `boundary_high_load_current_lookup_guarded_adaptive.csv`: compact 24-row matrix.
- `adaptive_vs_baselines.csv`: adaptive deltas against current, lookup, and guarded.
- `frontier_table.csv` and `results_by_workload_load_policy_scale.csv`: full aggregate summaries from the frontier summarizer.
- `current_vs_slo_no_preemption_adaptive_guarded.csv`: current-vs-adaptive comparison from the summarizer.
- `miss_by_shape.csv`: shape-level miss and latency summary.
- `bucket_size_distribution.csv` and `bucket_summary_by_policy.csv`: denoise bucket distributions and derived bucket summaries.
- `stagepool_replica_distribution.csv`: selected StagePool replica distribution.
- `figures/*.svg`: miss, goodput, throughput, P95, and mean-bucket figures.
- `omitted_artifacts_manifest.csv`: omitted raw/log/jsonl/trace/profile/benchmark_result artifacts from the raw OUTDIR.
