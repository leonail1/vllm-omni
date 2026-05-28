# E2E SLO Throughput Frontier

| Workload | Interarrival | Policy | SLO scale | Repeats | Miss rate | Goodput | Throughput | P95 latency | Mean bucket | P5 TTD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| large-heavy | 2.50s | current | 3 | 1 | 73.33% | 0.0401 | 0.1503 | 217.98s | 1.350 | -154024.6ms |
| large-heavy | 2.50s | slo_no_preemption_adaptive_guarded | 3 | 1 | 43.33% | 0.0895 | 0.1579 | 237.68s | 2.062 | -166533.2ms |
| large-heavy | 2.50s | slo_no_preemption_guarded | 3 | 1 | 55.00% | 0.0831 | 0.1847 | 191.48s | 1.797 | -134444.7ms |
| large-heavy | 2.50s | slo_no_preemption_lookup | 3 | 1 | 58.33% | 0.0570 | 0.1300 | 274.10s | 1.806 | -184135.0ms |
| large-heavy | 2.50s | slo_no_preemption_shape_guarded | 3 | 1 | 48.33% | 0.0888 | 0.1719 | 232.84s | 2.035 | -141453.3ms |
| rectangular-mix | 2.50s | current | 3 | 1 | 58.33% | 0.0597 | 0.1434 | 240.30s | 1.053 | -187945.8ms |
| rectangular-mix | 2.50s | slo_no_preemption_adaptive_guarded | 3 | 1 | 56.67% | 0.0961 | 0.2218 | 122.03s | 1.364 | -64202.1ms |
| rectangular-mix | 2.50s | slo_no_preemption_guarded | 3 | 1 | 55.00% | 0.1016 | 0.2258 | 158.41s | 1.261 | -100175.7ms |
| rectangular-mix | 2.50s | slo_no_preemption_lookup | 3 | 1 | 46.67% | 0.0993 | 0.1862 | 195.30s | 1.424 | -141393.1ms |
| rectangular-mix | 2.50s | slo_no_preemption_shape_guarded | 3 | 1 | 45.00% | 0.1159 | 0.2108 | 144.41s | 1.412 | -80227.0ms |

## Current vs slo_no_preemption_shape_guarded

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|
| large-heavy | 2.50s | 3 | 25.00 pp | 121.63% | 14.39% | 6.82% | +0.685 |
| rectangular-mix | 2.50s | 3 | 13.33 pp | 94.04% | 47.00% | -39.90% | +0.359 |

## Frontier by Miss Threshold

| Workload | Policy | SLO scale | Miss threshold | Selected interarrival | Throughput | Goodput | Miss rate | P95 latency | Mean bucket |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| large-heavy | current | 3 | 5% | none | | | | | |
| large-heavy | current | 3 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_adaptive_guarded | 3 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_adaptive_guarded | 3 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_guarded | 3 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_guarded | 3 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_lookup | 3 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_lookup | 3 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_shape_guarded | 3 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_shape_guarded | 3 | 10% | none | | | | | |
| rectangular-mix | current | 3 | 5% | none | | | | | |
| rectangular-mix | current | 3 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 3 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 3 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_guarded | 3 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_guarded | 3 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_lookup | 3 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_lookup | 3 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_shape_guarded | 3 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_shape_guarded | 3 | 10% | none | | | | | |
