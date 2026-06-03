# First-Stage Token SLO Smoke Report

This report is a minimal liveness check for the split first-stage DiT token SLO implementation. It is not a throughput-frontier result.

- `slo_no_preemption_token_objective`: completed 2/2 requests, failed 0, SLO miss rate 0.0%, P95 latency 14.08s.
- `slo_token_step_preemptive`: completed 2/2 requests, failed 0, SLO miss rate 0.0%, P95 latency 14.43s.

No PARD/DAG runtime, encoder DAG, or decoder DAG claim is made by this smoke package.
