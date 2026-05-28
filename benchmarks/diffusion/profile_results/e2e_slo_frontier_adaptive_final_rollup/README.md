# Adaptive DiT SLO Scheduler Final Rollup

This package is a GitHub-safe rollup of already committed result packages. It reads only committed package CSV/README material under `benchmarks/diffusion/profile_results/` and does not copy runtime output, logs, traces, raw profiles, JSONL, or large artifacts.

## Recommendation

Use `slo_no_preemption_adaptive_guarded` as the current candidate policy.

Do not promote `slo_no_preemption_shape_guarded`. The shape-aware area-threshold pilots are useful diagnostics, but they are not stable improvements: the default `0.6` guard helps `rectangular-mix` scale3 miss while regressing `large-heavy`, and the `0.5` threshold still does not fix `large-heavy` miss regression.

Keep the scheduler invariants unchanged: no step-level preemption, and continue using the lookup cost model from `benchmarks/diffusion/profile_results/latent_token_aspect/step_cost_model.json`.

## Evidence Summary

Normal-load evidence remains favorable enough for adaptive. In the bursty normal-load scale4 repeat0-2 package, adaptive has the best miss and P95 among the compared policies: adaptive miss `8.9%` versus lookup `13.3%`, current `20.0%`, and guarded `17.8%`.

High-load scale4/6 caveat repeat0-2 validation is the strongest current positive evidence. Across `large-heavy` and `rectangular-mix` at scales 4 and 6, adaptive has the best miss/goodput/P95 in all four repeated regions. The main caveat is packing pressure: `large-heavy scale4` still has bucket>=3 `37.6%`, versus lookup `20.8%` and guarded `27.9%`.

| Region | Adaptive Miss | Lookup Miss | Guarded Miss | Adaptive Goodput | Adaptive Throughput | Adaptive P95 | Adaptive Bucket>=3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| large-heavy 4.0 | 44.4% | 45.6% | 49.4% | 0.0875 | 0.1577 | 231.9s | 37.6% |
| large-heavy 6.0 | 29.4% | 32.2% | 37.2% | 0.1331 | 0.1886 | 194.5s | 29.5% |
| rectangular-mix 4.0 | 36.7% | 43.9% | 39.4% | 0.1264 | 0.1991 | 165.4s | 7.6% |
| rectangular-mix 6.0 | 19.4% | 26.1% | 22.8% | 0.1770 | 0.2197 | 138.1s | 6.3% |

Scale3/high-load remains a boundary rather than a solved production-no-miss setting. In repeat0-2 combined results, `large-heavy scale3` adaptive improves miss to `47.2%` versus guarded `53.9%` and lookup `57.8%`, but that is still a high miss region with high bucket pressure. `rectangular-mix scale3` adaptive has better latency/throughput but trails lookup/guarded on miss.

## Shape Guard Decision

The shape guard pilots are not recommended as the final policy path.

| Guard/Workload | Adaptive Miss | Shape Guard Miss | Adaptive P95 | Shape Guard P95 | Adaptive Bucket>=3 | Shape Guard Bucket>=3 | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| 0.6_default large-heavy | 43.3% | 48.3% | 237.7s | 232.8s | 38.8% | 33.3% | not recommended because large-heavy miss regresses versus adaptive |
| 0.6_default rectangular-mix | 56.7% | 45.0% | 122.0s | 144.4s | 1.2% | 5.6% | useful rectangular signal but not enough to offset large-heavy regression and P95 tradeoff |
| 0.5_threshold large-heavy | 41.7% | 50.0% | 236.7s | 179.1s | 38.4% | 28.7% | does not fix large-heavy miss regression; mostly trades miss for P95/throughput/bucket |
| 0.5_threshold rectangular-mix | 48.3% | 48.3% | 135.8s | 136.2s | 7.6% | 9.9% | rectangular miss is tied, with small goodput/throughput gain and no P95 win |

Important shape-level notes:

- Default `0.6` shape guard improves `large-heavy` 1024x1024 misses but regresses 768x768 heavily and leaves 512x512 fully missed in the pilot.
- Threshold `0.5` does not repair `large-heavy`; 1024x1024 and 768x768 both miss more often than adaptive in that ablation.
- `rectangular-mix` still has a 1024x1024 tradeoff in repeated scale6 evidence: adaptive improves 512x768 and 512x512 versus guarded, but 1024x1024 is worse (`6/24` misses versus guarded `4/24`).

## Residual Risks

- `large-heavy` bucket>=3 remains high under adaptive, especially at high load; this is the clearest remaining packing-pressure signal.
- High-load scale3 remains a boundary condition with substantial miss, not a production no-miss proof.
- Rectangular and shape-level outcomes still include tradeoffs across 1024x1024 and smaller rectangular shapes.
- Some boundary evidence is repeat0 only (`bursty` scale3/3.5 and the scale3.5 sweep); the repeat-validated evidence is strongest for scale3 large/rectangular and scale4/6 large/rectangular.

## Files

- `source_packages.csv`: source packages, commit status, review status inferred from the goal thread, and usage.
- `adaptive_high_load_summary.csv`: high-load scale3/3.5/4/6 adaptive comparison against current/lookup/guarded.
- `repeat_validated_caveats.csv`: repeat0-2 combined scale4/6 caveat regions.
- `shape_guard_ablation_summary.csv`: default `0.6` and threshold `0.5` shape guard outcomes and decision notes.
- `recommendation_summary.csv`: final policy recommendation, rejection of shape guard, and next action.
- `normal_load_context.csv`: compact normal-load bursty repeat context.
- SVG figures: `adaptive_high_load_miss_rank.svg`, `caveat_combined_miss.svg`, `shape_guard_ablation_tradeoff.svg`.
- `omitted_artifacts_manifest.csv`: sentinel manifest for this rollup-only package.

## Next Step

Do not add a global conservative cap and do not keep tuning the area-threshold shape guard by default. The next useful work is a final PR/commit explanation and lightweight documentation or cleanup around `slo_no_preemption_adaptive_guarded`, while preserving the current no-preemption and lookup-model assumptions.
