# E2E SLO Throughput Frontier

| Workload | Interarrival | Policy | SLO scale | Repeats | Miss rate | Goodput | Throughput | P95 latency | Mean bucket | P5 TTD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| large-heavy | 2.50s | slo_no_preemption_adaptive_guarded | 3 | 1 | 41.67% | 0.0942 | 0.1615 | 236.67s | 2.099 | -165900.8ms |
| large-heavy | 2.50s | slo_no_preemption_shape_guarded | 3 | 1 | 50.00% | 0.1011 | 0.2022 | 179.12s | 1.928 | -114542.2ms |
| rectangular-mix | 2.50s | slo_no_preemption_adaptive_guarded | 3 | 1 | 48.33% | 0.1026 | 0.1986 | 135.82s | 1.410 | -72115.6ms |
| rectangular-mix | 2.50s | slo_no_preemption_shape_guarded | 3 | 1 | 48.33% | 0.1071 | 0.2072 | 136.16s | 1.395 | -78146.7ms |

## Current vs slo_no_preemption_shape_guarded

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|

## Frontier by Miss Threshold

| Workload | Policy | SLO scale | Miss threshold | Selected interarrival | Throughput | Goodput | Miss rate | P95 latency | Mean bucket |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| large-heavy | slo_no_preemption_adaptive_guarded | 3 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_adaptive_guarded | 3 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_shape_guarded | 3 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_shape_guarded | 3 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 3 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 3 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_shape_guarded | 3 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_shape_guarded | 3 | 10% | none | | | | | |
