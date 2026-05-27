# E2E SLO Throughput Frontier

| Workload | Interarrival | Policy | SLO scale | Repeats | Miss rate | Goodput | Throughput | P95 latency | Mean bucket | P5 TTD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current-mix | 2.50s | slo_no_preemption_adaptive_guarded | 6 | 2 | 11.67% | 0.2109 | 0.2387 | 133.97s | 1.718 | -17331.6ms |
| current-mix | 2.50s | slo_no_preemption_guarded | 6 | 2 | 15.00% | 0.1727 | 0.2032 | 157.50s | 1.631 | -37918.0ms |
| current-mix | 4.25s | slo_no_preemption_adaptive_guarded | 6 | 2 | 0.00% | 0.2029 | 0.2029 | 56.87s | 1.121 | 57501.5ms |
| current-mix | 4.25s | slo_no_preemption_guarded | 6 | 2 | 0.00% | 0.2102 | 0.2102 | 60.42s | 1.122 | 61323.2ms |

## Current vs slo_no_preemption_adaptive_guarded

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|

## Frontier by Miss Threshold

| Workload | Policy | SLO scale | Miss threshold | Selected interarrival | Throughput | Goodput | Miss rate | P95 latency | Mean bucket |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current-mix | slo_no_preemption_adaptive_guarded | 6 | 5% | 4.25s | 0.2029 | 0.2029 | 0.00% | 56.87s | 1.121 |
| current-mix | slo_no_preemption_adaptive_guarded | 6 | 10% | 4.25s | 0.2029 | 0.2029 | 0.00% | 56.87s | 1.121 |
| current-mix | slo_no_preemption_guarded | 6 | 5% | 4.25s | 0.2102 | 0.2102 | 0.00% | 60.42s | 1.122 |
| current-mix | slo_no_preemption_guarded | 6 | 10% | 4.25s | 0.2102 | 0.2102 | 0.00% | 60.42s | 1.122 |
