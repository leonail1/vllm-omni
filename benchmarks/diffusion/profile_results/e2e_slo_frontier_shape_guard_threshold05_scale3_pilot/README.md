# Shape Guard Threshold 0.5 Scale3 Pilot

This directory contains a lightweight, GitHub-safe result bundle for the threshold=0.5 shape-guard ablation of the no-preemption DiT SLO scheduler. It intentionally contains only the adaptive vs shape_guarded small matrix for large-heavy and rectangular-mix at ia=2.5, scale=3.0, repeat=0. Runtime logs, JSON outputs, raw step profiles, stagepool profiles, benchmark_result files, and trace text/manifests are intentionally omitted and listed in `omitted_artifacts_manifest.csv`.

This is a fixed-load, fixed-scale ablation of `SHAPE_GUARDED_SMALL_SHAPE_MAX_AREA_RATIO=0.5`; it does not sweep offered load or SLO scale.

## Experiment Matrix

- Workloads: `large-heavy, rectangular-mix`
- Interarrival seconds: `2.5`
- SLO scales: `3`
- Repeats: `0`
- Policies: `slo_no_preemption_adaptive_guarded, slo_no_preemption_shape_guarded`
- Candidate policy: `slo_no_preemption_shape_guarded`
- Shape guard area threshold: `0.5`
- Shape guard queue window: `20000 ms`
- Shape guard max queue length: `4`
- Step preemption: disabled for both policies
- Cost model: `benchmarks/diffusion/profile_results/latent_token_aspect/step_cost_model.json`

## Command

```bash
cd /home/lzg/vllm-omni
OUTDIR=/home/lzg/vllm-omni/benchmark_outputs/e2e_slo_frontier_shape_guard_threshold05_scale3_pilot PROFILE_MODE=e2e_slo_frontier_shape_guard_threshold05_scale3_pilot WORKLOADS=large-heavy,rectangular-mix INTERARRIVALS=2.5 SCALES=3.0 REPEATS=0 POLICIES=slo_no_preemption_adaptive_guarded,slo_no_preemption_shape_guarded CANDIDATE_POLICY=slo_no_preemption_shape_guarded SHAPE_GUARDED_SMALL_SHAPE_MAX_AREA_RATIO=0.5 SHAPE_GUARDED_SMALL_SHAPE_QUEUE_GUARD_WINDOW_MS=20000 SHAPE_GUARDED_SMALL_SHAPE_QUEUE_GUARD_MAX_QUEUE_LENGTH=4 PORT_BASE=22820 MASTER_PORT_BASE=30820 bash benchmarks/diffusion/run_910b_e2e_slo_frontier.sh
```

## Conclusion

Threshold=0.5 does not fix the `large-heavy` scale-3 miss regression. `slo_no_preemption_shape_guarded` has worse miss rate than adaptive on `large-heavy` (50.0% vs 41.7%), including worse 1024x1024 miss (12/36 vs 10/36) and 768x768 miss (7/12 vs 4/12), while 512x512 remains 11/12 for both. The benefit is a different tradeoff: lower P95 (179.1s vs 236.7s), higher throughput (0.2022 vs 0.1615), and lower bucket>=3 share (28.7% vs 38.4%).

On `rectangular-mix`, threshold=0.5 is essentially miss-neutral: both policies miss 48.3%. Shape-guarded has slightly higher goodput and throughput, nearly identical P95 (136.2s vs 135.8s), and a higher bucket>=3 share (9.9% vs 7.6%).

Practical read: threshold=0.5 is useful as a negative ablation. It confirms that narrowing the area threshold alone shifts the latency/packing tradeoff but does not recover the large-heavy SLO miss behavior, so it should not be expanded as the final candidate without another mechanism.

## Row Table

| Workload | Policy | Miss | Goodput | Throughput | P95 | Bucket>=3 |
|---|---|---:|---:|---:|---:|---:|
| large-heavy | slo_no_preemption_adaptive_guarded | 41.7% | 0.0942 | 0.1615 | 236.7s | 38.4% |
| large-heavy | slo_no_preemption_shape_guarded | 50.0% | 0.1011 | 0.2022 | 179.1s | 28.7% |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 48.3% | 0.1026 | 0.1986 | 135.8s | 7.6% |
| rectangular-mix | slo_no_preemption_shape_guarded | 48.3% | 0.1071 | 0.2072 | 136.2s | 9.9% |

## Current vs slo_no_preemption_shape_guarded

`current` was intentionally not run in this narrow ablation, so the generated current-vs-candidate table is empty.

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|

## Files

- `frontier_table.csv`: aggregate policy x workload x load x scale table.
- `results_by_workload_load_policy_scale.csv`: same aggregate table with a descriptive filename.
- `frontier_by_threshold.csv`: best throughput under configured miss-rate thresholds.
- `current_vs_slo_no_preemption_shape_guarded.csv`: generated comparison file; empty for this two-policy ablation because `current` was intentionally not run.
- `miss_by_shape.csv`: shape-level miss rate and latency.
- `bucket_size_distribution.csv`: denoise bucket size counts from step profiles.
- `stagepool_replica_distribution.csv`: selected replica counts from StagePool profiles.
- `frontier_report.md`: compact markdown report generated from the raw output directory.
- `figures/*.svg`: compact visualizations for miss rate, goodput, throughput, P95 latency, and mean bucket size.
- `omitted_artifacts_manifest.csv`: omitted runtime/raw artifacts.

No JSON, raw trace/profile, log, or benchmark_result files are included in this package.
