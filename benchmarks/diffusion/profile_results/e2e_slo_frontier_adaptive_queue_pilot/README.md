# E2E SLO Frontier Adaptive Queue Pilot

GitHub-safe result bundle for the adaptive StagePool queue guard pilot. Runtime logs, raw step profiles, StagePool JSONL, trace text, and PID files are intentionally omitted and listed in `omitted_artifacts_manifest.csv`.

## Matrix

- Workloads: `current-mix`, `shape-grouped-current-mix`
- Interarrival: `2.5s`
- SLO scales: `4.0`, `6.0`
- Repeat: `0`
- Policies: `current`, `slo_no_preemption_lookup`, `slo_no_preemption_guarded`, `slo_no_preemption_adaptive_guarded`

## Notes

The adaptive policy uses the guarded no-preemption lookup scheduler plus a StagePool queue guard (`stagepool_queue_guard_window_ms=10000`, `stagepool_queue_guard_max_queue_length=4`). It fixes the target current-mix scale=6 miss regression in this pilot, while shape-grouped scale=6 has a small one-request miss regression versus guarded.

## Files

- `results_by_workload_load_policy_scale.csv`: aggregate metrics.
- `current_lookup_guarded_vs_adaptive.csv`: deltas against current/lookup/guarded.
- `miss_by_shape.csv`: shape-level miss and P95 latency.
- `bucket_size_distribution.csv`: filtered denoise bucket counts.
- `stagepool_replica_distribution.csv`: selected replica counts.
- `figures/*.svg`: compact metric charts.
- `omitted_artifacts_manifest.csv`: runtime/raw artifacts intentionally excluded from git.
