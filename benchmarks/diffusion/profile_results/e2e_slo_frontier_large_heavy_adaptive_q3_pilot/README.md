# large-heavy adaptive q3 pilot: failure/partial package

Status: incomplete. The main pilot failed after starting `repeat=2, scale=4.0`; this package intentionally includes only the clean completed partial data from `repeat=1` at SLO scales 4 and 6.

Included matrix slice:

- Workload: `large-heavy`
- Interarrival: `2.5s`
- Policy: `slo_no_preemption_adaptive_guarded`
- Variant: `ADAPTIVE_STAGEPOOL_QUEUE_GUARD_MAX_QUEUE_LENGTH=3`, window `10000ms`
- Included repeats: `1`
- Expected repeats: `1,2`
- Missing runs: `1` per scale (`repeat=2` excluded/missing for partial reporting)

Failure note:

- Runner reached `repeat=2, scale=4.0` and then marked the policy failed at `2026-05-31 16:48:29`.
- `repeat_2/scale_4/client.log` is empty; the server log shows shutdown followed by repeated Ascend TBE repository-manager `Thread-2` `EOFError` tracebacks.
- No vLLM or benchmark process remained after cleanup. A standalone `tmux` process existed and was not touched.

Files:

- `key_metrics.csv`: per-run partial metrics for repeat 1.
- `partial_metrics.csv`: per-scale partial summary with `incomplete=true` and `missing_runs=1`.
- `default_adaptive_delta_partial.csv`: q3 repeat1 partial values compared with committed default-adaptive repeat1-2 means.
- `failure_summary.csv`: concise failure context.
- `omitted_artifacts_manifest.csv`: raw/runtime/result artifacts deliberately omitted from this GitHub-safe package.
- `figures/partial_miss_bucket_ge3.svg`: compact partial visualization.

This package deliberately contains no raw logs, JSON/JSONL files, traces, status files, or benchmark result payloads.
