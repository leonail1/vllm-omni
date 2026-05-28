# E2E SLO Throughput Frontier

| Workload | Interarrival | Policy | SLO scale | Repeats | Miss rate | Goodput | Throughput | P95 latency | Mean bucket | P5 TTD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| large-heavy | 2.50s | slo_no_preemption_adaptive_guarded | 4 | 2 | 44.17% | 0.0896 | 0.1609 | 231.43s | 2.041 | -142873.7ms |
| large-heavy | 2.50s | slo_no_preemption_adaptive_guarded | 6 | 2 | 32.50% | 0.1279 | 0.1893 | 182.70s | 1.888 | -61602.7ms |
| large-heavy | 2.50s | slo_no_preemption_guarded | 4 | 2 | 47.50% | 0.0882 | 0.1682 | 218.56s | 1.838 | -120256.8ms |
| large-heavy | 2.50s | slo_no_preemption_guarded | 6 | 2 | 36.67% | 0.0924 | 0.1461 | 248.99s | 1.777 | -93404.4ms |
| large-heavy | 2.50s | slo_no_preemption_lookup | 4 | 2 | 47.50% | 0.0822 | 0.1476 | 244.15s | 1.820 | -142258.2ms |
| large-heavy | 2.50s | slo_no_preemption_lookup | 6 | 2 | 32.50% | 0.1102 | 0.1589 | 247.05s | 1.776 | -75251.0ms |
| rectangular-mix | 2.50s | slo_no_preemption_adaptive_guarded | 4 | 2 | 35.83% | 0.1379 | 0.2149 | 138.90s | 1.356 | -63553.2ms |
| rectangular-mix | 2.50s | slo_no_preemption_adaptive_guarded | 6 | 2 | 17.50% | 0.1807 | 0.2189 | 128.01s | 1.283 | -7359.5ms |
| rectangular-mix | 2.50s | slo_no_preemption_guarded | 4 | 2 | 38.33% | 0.1208 | 0.1941 | 169.01s | 1.371 | -92816.5ms |
| rectangular-mix | 2.50s | slo_no_preemption_guarded | 6 | 2 | 25.00% | 0.1662 | 0.2215 | 155.95s | 1.388 | -41985.7ms |
| rectangular-mix | 2.50s | slo_no_preemption_lookup | 4 | 2 | 42.50% | 0.1071 | 0.1864 | 169.98s | 1.373 | -94741.4ms |
| rectangular-mix | 2.50s | slo_no_preemption_lookup | 6 | 2 | 28.33% | 0.1552 | 0.2134 | 167.75s | 1.369 | -43833.9ms |

## Current vs slo_no_preemption_adaptive_guarded

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|

## Frontier by Miss Threshold

| Workload | Policy | SLO scale | Miss threshold | Selected interarrival | Throughput | Goodput | Miss rate | P95 latency | Mean bucket |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| large-heavy | slo_no_preemption_adaptive_guarded | 4 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_adaptive_guarded | 4 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_adaptive_guarded | 6 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_adaptive_guarded | 6 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_guarded | 4 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_guarded | 4 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_guarded | 6 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_guarded | 6 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_lookup | 4 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_lookup | 4 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_lookup | 6 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_lookup | 6 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 4 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 4 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 6 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 6 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_guarded | 4 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_guarded | 4 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_guarded | 6 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_guarded | 6 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_lookup | 4 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_lookup | 4 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_lookup | 6 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_lookup | 6 | 10% | none | | | | | |
