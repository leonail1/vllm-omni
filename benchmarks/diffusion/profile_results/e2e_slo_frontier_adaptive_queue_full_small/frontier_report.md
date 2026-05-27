# E2E SLO Throughput Frontier

| Workload | Interarrival | Policy | SLO scale | Repeats | Miss rate | Goodput | Throughput | P95 latency | Mean bucket | P5 TTD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current-mix | 2.50s | slo_no_preemption_adaptive_guarded | 4 | 1 | 40.00% | 0.0876 | 0.1460 | 248.91s | 2.020 | -148292.3ms |
| current-mix | 2.50s | slo_no_preemption_adaptive_guarded | 6 | 1 | 36.67% | 0.1292 | 0.2040 | 204.43s | 1.874 | -64889.8ms |
| current-mix | 4.25s | slo_no_preemption_adaptive_guarded | 4 | 1 | 6.67% | 0.1511 | 0.1619 | 85.33s | 1.137 | 8422.9ms |
| current-mix | 4.25s | slo_no_preemption_adaptive_guarded | 6 | 1 | 61.67% | 0.0495 | 0.1183 | 290.70s | 2.238 | -169062.1ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_adaptive_guarded | 4 | 1 | 18.33% | 0.2079 | 0.2545 | 94.24s | 1.829 | -16087.4ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_adaptive_guarded | 6 | 1 | 6.67% | 0.2322 | 0.2488 | 100.68s | 1.914 | 9370.6ms |
| shape-grouped-current-mix | 4.25s | slo_no_preemption_adaptive_guarded | 4 | 1 | 0.00% | 0.2169 | 0.2169 | 26.73s | 1.093 | 47565.0ms |
| shape-grouped-current-mix | 4.25s | slo_no_preemption_adaptive_guarded | 6 | 1 | 0.00% | 0.2166 | 0.2166 | 26.59s | 1.087 | 80905.7ms |

## Current vs slo_no_preemption_adaptive_guarded

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|

## Frontier by Miss Threshold

| Workload | Policy | SLO scale | Miss threshold | Selected interarrival | Throughput | Goodput | Miss rate | P95 latency | Mean bucket |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current-mix | slo_no_preemption_adaptive_guarded | 4 | 5% | none | | | | | |
| current-mix | slo_no_preemption_adaptive_guarded | 4 | 10% | 4.25s | 0.1619 | 0.1511 | 6.67% | 85.33s | 1.137 |
| current-mix | slo_no_preemption_adaptive_guarded | 6 | 5% | none | | | | | |
| current-mix | slo_no_preemption_adaptive_guarded | 6 | 10% | none | | | | | |
| shape-grouped-current-mix | slo_no_preemption_adaptive_guarded | 4 | 5% | 4.25s | 0.2169 | 0.2169 | 0.00% | 26.73s | 1.093 |
| shape-grouped-current-mix | slo_no_preemption_adaptive_guarded | 4 | 10% | 4.25s | 0.2169 | 0.2169 | 0.00% | 26.73s | 1.093 |
| shape-grouped-current-mix | slo_no_preemption_adaptive_guarded | 6 | 5% | 4.25s | 0.2166 | 0.2166 | 0.00% | 26.59s | 1.087 |
| shape-grouped-current-mix | slo_no_preemption_adaptive_guarded | 6 | 10% | 2.50s | 0.2488 | 0.2322 | 6.67% | 100.68s | 1.914 |
