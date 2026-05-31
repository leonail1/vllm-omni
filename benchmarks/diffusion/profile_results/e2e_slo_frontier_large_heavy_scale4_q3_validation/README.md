# large-heavy scale4 q3 validation

GitHub-safe result package for the large-heavy scale4 q3 validation run on 910B.

Run command:

```bash
cd /home/lzg/vllm-omni && env \
  OUTDIR=/home/lzg/vllm-omni/benchmark_outputs/e2e_slo_frontier_large_heavy_scale4_q3_validation \
  WORKLOADS=large-heavy \
  INTERARRIVALS=2.5 \
  SCALES=4.0 \
  REPEATS=2 \
  POLICIES=slo_no_preemption_adaptive_guarded \
  CANDIDATE_POLICY=slo_no_preemption_adaptive_guarded \
  ADAPTIVE_STAGEPOOL_QUEUE_GUARD_MAX_QUEUE_LENGTH=3 \
  ADAPTIVE_STAGEPOOL_QUEUE_GUARD_WINDOW_MS=10000 \
  SKIP_EXISTING=1 \
  bash benchmarks/diffusion/run_910b_e2e_slo_frontier.sh
```

Sources:

- q3 repeat1 top-line metrics: `benchmarks/diffusion/profile_results/e2e_slo_frontier_large_heavy_adaptive_q3_pilot/key_metrics.csv`.
- q3 repeat2 validation metrics: `/home/lzg/vllm-omni/benchmark_outputs/e2e_slo_frontier_large_heavy_scale4_q3_validation`.
- default adaptive comparator: `benchmarks/diffusion/profile_results/e2e_slo_frontier_high_load_scale46_caveat_repeats_adaptive/mean_by_policy_repeat1_2.csv`, large-heavy scale4 repeat1-2 mean.

Result status:

- q3 repeat2 completed `60/60` requests and wrote `benchmark_result.json` plus frontier summary.
- Runner status completed at `2026-05-31 17:30:01`.
- Server log had shutdown-time `StageDiffusionProc died unexpectedly` / `EOFError` noise after request completion; no benchmark or vLLM process remained after cleanup.

Top-line repeat1-2 mean:

- Miss rate: q3 43.33% vs default 44.17% (-0.83 pp).
- Goodput: q3 0.091108 rps vs default 0.089646 rps (+1.63%).
- Throughput: q3 0.160779 rps vs default 0.160910 rps (-0.08%).
- P95 latency: q3 231.532s vs default 231.432s (+0.04%).
- Mean bucket size: q3 1.853 vs default 2.041 (-0.188).
- Bucket >=3 rate: q3 26.56% vs default 35.88% (-9.33 pp).

Shape-level miss, combined repeat1-2:

- 1024x1024: q3 30.56% vs default 34.72% (-4.17 pp)
- 512x512: q3 45.83% vs default 83.33% (-37.50 pp)
- 768x768: q3 79.17% vs default 33.33% (+45.83 pp)


Notes on aggregation:

- q3 repeat1 top-line metrics come from the committed partial package, while repeat1 shape/bucket/stagepool drilldowns are derived from the corresponding benchmark output profiles and are included only as GitHub-safe CSV summaries.
- `key_metrics.csv` reports repeat1-2 arithmetic means. The combined rows in bucket and replica distribution CSVs pool profile-row counts across repeats, so their percentages can differ slightly from repeat-mean percentages.

Files:

- `key_metrics.csv`: compact q3 vs default metric table.
- `combined_q3_repeat1_2.csv`: q3 repeat1, repeat2, and repeat1-2 mean.
- `default_adaptive_delta.csv`: wide q3-minus-default comparison row.
- `miss_by_shape.csv`: per-repeat and combined shape-level miss data.
- `bucket_size_distribution.csv`: per-repeat and combined step bucket distributions.
- `stagepool_replica_distribution.csv`: per-repeat and combined stagepool selected replica distributions.
- `figures/*.svg`: GitHub-safe SVG figures.
- `omitted_artifacts_manifest.csv`: raw artifacts deliberately omitted.

This package deliberately contains no raw logs, JSON/JSONL files, traces, status files, benchmark result payloads, or large runtime profiles.
