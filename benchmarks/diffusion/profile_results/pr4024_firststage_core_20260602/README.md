# PR4024 First-stage Token SLO Results

This GitHub-safe bundle summarizes the first-stage PR4024 DiT token scheduler experiments on the 910B machine. It intentionally omits runtime logs, raw JSONL profiles, traces, and large artifacts.

## Scope

- Model: Qwen-Image, int8.
- Hardware/runtime: 8 DiT replicas, TP1, one NPU per replica, devices 0..7.
- Matrix: workloads `current-mix` and `shape-grouped-current-mix`; interarrival 2.5s and 4.25s; SLO scale 4.0 and 6.0; repeat 0; 20 requests per cell.
- Baseline `current` was rerun with `TORCH_SDPA/SDPA` after the original FLASH_ATTN diagnostic baseline hit static-attention failures.
- Token policies use PR4024 dynamic token batching with `FLASH_ATTN/MindIE-SD`.

## Policy names

- `current`: vLLM-Omni current scheduler baseline after the static-attention backend fix.
- `pr4024_dynamic`: PR4024 heterogeneous diffusion dynamic batching without SLO-aware admission.
- `slo_no_preemption_token_guarded`: token-level SLO scheduler with conservative safe admission and no step-level preemption.
- `slo_no_preemption_token_adaptive`: adaptive token-level SLO variant; it reduces latency but had failed requests in this run.
- `slo_no_preemption_token_objective`: token-level SLO scheduler using the StagePool target function/objective.
- `slo_token_step_preemptive`: token-level SLO scheduler that allows step-boundary preemption.
- `current_diagnostic_flash_attn`: original failed current baseline kept only as diagnostic evidence; do not use as the main baseline.

## Main conclusion

- `slo_no_preemption_token_objective` completed 160/160 requests with 0 failures and 0 successful-request SLO misses. Compared with fixed `current`, mean goodput improves by 7.3% and mean P95 latency changes by -32.2%.
- `slo_no_preemption_token_guarded` also completed 160/160 requests with no failures or SLO misses; mean goodput improves by 6.4% vs fixed `current`.
- `slo_token_step_preemptive` completed 160/160 requests with 0 failures and mean goodput 0.250; compare with no-preemption policies before choosing it.

## Files

- `summary_by_policy.csv`: aggregate metrics by policy.
- `results_by_cell.csv`: per workload/interarrival/scale/policy metrics.
- `comparison_vs_current.csv`: deltas against the fixed `current` baseline.
- `shape_outcomes.csv`: request outcomes grouped by shape.
- `experiment_settings.csv`: matrix and output-dir metadata.
- `figures/*.svg`: compact figures for goodput, throughput, P95 latency, failure rate, and SLO miss rate.
- `omitted_artifacts_manifest.csv`: raw artifacts intentionally omitted from the safe bundle.
