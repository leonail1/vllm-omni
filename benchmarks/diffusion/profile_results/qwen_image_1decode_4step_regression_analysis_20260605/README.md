# Qwen-Image 1decode 4-step Regression Analysis

正式 pipeline 口径固定为 `1decode`：NPU0 负责 encode + 1 个 decode replica，NPU1-7 负责 7 个 DiT replica。
本包只比较 `baseline_7replica` 与 `pipeline_1decode`，dummy/stagger 是诊断 cell，不作为正式候选方案。

## Main 4-step Snapshot

| label | throughput(img/s) | P95 latency(s) | decode queue P95(s) |
|---|---:|---:|---:|
| baseline_7replica | 1.988 | 21.069 | 0.000 |
| pipeline_1decode | 1.762 | 23.371 | 2.988 |
| pipeline_1decode_dummy_decode | 2.066 | 20.226 | 0.018 |
| pipeline_1decode_stagger_0p15 | 1.846 | 17.860 | 2.209 |

## Trace Additions

- `*_payload_export`: upstream stage output is detached/copied to CPU for cross-stage handoff.
- `*_payload_import`: downstream stage receives the payload and moves tensors to its local device.
- `dit_completion_events.csv`: processed per-request DiT completion wall timestamps and decode queue delay.

## Transfer Rows

See `transfer_overhead_summary.csv` for payload import/export timing.

## Files

- `README.md`: summary and interpretation.
- `run_status.csv`: run status and output locations.
- `scenario_comparison.csv`: throughput and latency by scenario.
- `stage_timing_breakdown.csv`: stage and transfer event timing.
- `transfer_overhead_summary.csv`: payload handoff overhead.
- `decode_queue_summary.csv`: submit-to-decode queue delay.
- `dit_completion_burst_summary.csv`: DiT completion burst metrics.
- `dit_completion_events.csv`: processed per-request DiT completion timestamps.
- `route_summary.csv`: stage/replica request distribution.
- `npu_summary.csv`: summarized NPU resource samples.
- `npu_sampling_warnings.csv`: sampler warnings, empty when none were observed.
- `throughput_comparison.svg`, `p95_comparison.svg`, `decode_queue.svg`, `stage_breakdown.svg`: safe summary figures.
- `omitted_artifacts_manifest.md`: raw/log/jsonl/trace/profile omission record.
