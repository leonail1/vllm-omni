# E2E SLO Throughput Frontier

| Workload | Interarrival | Policy | SLO scale | Repeats | Miss rate | Goodput | Throughput | P95 latency | Mean bucket | P5 TTD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bursty | 2.50s | slo_no_preemption_adaptive_guarded | 4 | 2 | 31.67% | 0.1410 | 0.2063 | 150.68s | 1.741 | -59131.6ms |
| bursty | 2.50s | slo_no_preemption_guarded | 4 | 2 | 38.33% | 0.1183 | 0.1935 | 178.82s | 1.685 | -95019.6ms |
| large-heavy | 2.50s | slo_no_preemption_adaptive_guarded | 4 | 2 | 43.33% | 0.1128 | 0.1991 | 194.60s | 2.014 | -108056.0ms |
| large-heavy | 2.50s | slo_no_preemption_guarded | 4 | 2 | 50.00% | 0.0816 | 0.1631 | 223.68s | 1.819 | -127618.6ms |

## Current vs slo_no_preemption_adaptive_guarded

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|

## Frontier by Miss Threshold

| Workload | Policy | SLO scale | Miss threshold | Selected interarrival | Throughput | Goodput | Miss rate | P95 latency | Mean bucket |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bursty | slo_no_preemption_adaptive_guarded | 4 | 5% | none | | | | | |
| bursty | slo_no_preemption_adaptive_guarded | 4 | 10% | none | | | | | |
| bursty | slo_no_preemption_guarded | 4 | 5% | none | | | | | |
| bursty | slo_no_preemption_guarded | 4 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_adaptive_guarded | 4 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_adaptive_guarded | 4 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_guarded | 4 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_guarded | 4 | 10% | none | | | | | |
