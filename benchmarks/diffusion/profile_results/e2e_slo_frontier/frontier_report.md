# E2E SLO Throughput Frontier

| Workload | Interarrival | Policy | SLO scale | Repeats | Miss rate | Goodput | Throughput | P95 latency | Mean bucket | P5 TTD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current-mix | 2.50s | current | 4 | 1 | 45.00% | 0.1067 | 0.1941 | 156.36s | 1.257 | -75475.9ms |
| current-mix | 2.50s | current | 6 | 1 | 13.33% | 0.1748 | 0.2016 | 141.62s | 1.303 | -29301.2ms |
| current-mix | 2.50s | slo_no_preemption_lookup | 4 | 1 | 36.67% | 0.1009 | 0.1594 | 243.97s | 1.658 | -158100.2ms |
| current-mix | 2.50s | slo_no_preemption_lookup | 6 | 1 | 21.67% | 0.1518 | 0.1937 | 180.73s | 1.817 | -70700.1ms |
| current-mix | 4.25s | current | 4 | 1 | 31.67% | 0.1179 | 0.1726 | 102.07s | 1.231 | -23199.0ms |
| current-mix | 4.25s | current | 6 | 1 | 13.33% | 0.1273 | 0.1468 | 147.07s | 1.208 | -30339.9ms |
| current-mix | 4.25s | slo_no_preemption_lookup | 4 | 1 | 6.67% | 0.1906 | 0.2042 | 82.43s | 1.198 | -103.5ms |
| current-mix | 4.25s | slo_no_preemption_lookup | 6 | 1 | 0.00% | 0.1949 | 0.1949 | 65.67s | 1.115 | 50887.7ms |
| shape-grouped-current-mix | 2.50s | current | 4 | 1 | 31.67% | 0.1700 | 0.2488 | 97.75s | 1.829 | -21987.2ms |
| shape-grouped-current-mix | 2.50s | current | 6 | 1 | 8.33% | 0.2149 | 0.2345 | 128.57s | 1.838 | -18825.1ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_lookup | 4 | 1 | 35.00% | 0.1418 | 0.2181 | 145.14s | 1.993 | -65633.1ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_lookup | 6 | 1 | 13.33% | 0.2020 | 0.2330 | 130.74s | 2.042 | -23928.1ms |
| shape-grouped-current-mix | 4.25s | current | 4 | 1 | 0.00% | 0.2083 | 0.2083 | 57.08s | 1.572 | 23337.3ms |
| shape-grouped-current-mix | 4.25s | current | 6 | 1 | 1.67% | 0.1808 | 0.1838 | 100.77s | 1.567 | 18392.3ms |
| shape-grouped-current-mix | 4.25s | slo_no_preemption_lookup | 4 | 1 | 0.00% | 0.2166 | 0.2166 | 27.89s | 1.124 | 47173.2ms |
| shape-grouped-current-mix | 4.25s | slo_no_preemption_lookup | 6 | 1 | 0.00% | 0.2168 | 0.2168 | 27.12s | 1.111 | 80189.2ms |

## Current vs Candidate

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|
| current-mix | 2.50s | 4 | 8.33 pp | -5.42% | -17.87% | 56.03% | +0.402 |
| current-mix | 2.50s | 6 | -8.33 pp | -13.16% | -3.92% | 27.61% | +0.514 |
| current-mix | 4.25s | 4 | 25.00 pp | 61.66% | 18.36% | -19.24% | -0.033 |
| current-mix | 4.25s | 6 | 13.33 pp | 53.17% | 32.75% | -55.35% | -0.092 |
| shape-grouped-current-mix | 2.50s | 4 | -3.33 pp | -16.62% | -12.34% | 48.48% | +0.164 |
| shape-grouped-current-mix | 2.50s | 6 | -5.00 pp | -6.03% | -0.60% | 1.69% | +0.204 |
| shape-grouped-current-mix | 4.25s | 4 | 0.00 pp | 3.96% | 3.96% | -51.15% | -0.448 |
| shape-grouped-current-mix | 4.25s | 6 | 1.67 pp | 19.90% | 17.90% | -73.09% | -0.456 |

## Frontier by Miss Threshold

| Workload | Policy | SLO scale | Miss threshold | Selected interarrival | Throughput | Goodput | Miss rate | P95 latency | Mean bucket |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current-mix | current | 4 | 5% | none | | | | | |
| current-mix | current | 4 | 10% | none | | | | | |
| current-mix | current | 6 | 5% | none | | | | | |
| current-mix | current | 6 | 10% | none | | | | | |
| current-mix | slo_no_preemption_lookup | 4 | 5% | none | | | | | |
| current-mix | slo_no_preemption_lookup | 4 | 10% | 4.25s | 0.2042 | 0.1906 | 6.67% | 82.43s | 1.198 |
| current-mix | slo_no_preemption_lookup | 6 | 5% | 4.25s | 0.1949 | 0.1949 | 0.00% | 65.67s | 1.115 |
| current-mix | slo_no_preemption_lookup | 6 | 10% | 4.25s | 0.1949 | 0.1949 | 0.00% | 65.67s | 1.115 |
| shape-grouped-current-mix | current | 4 | 5% | 4.25s | 0.2083 | 0.2083 | 0.00% | 57.08s | 1.572 |
| shape-grouped-current-mix | current | 4 | 10% | 4.25s | 0.2083 | 0.2083 | 0.00% | 57.08s | 1.572 |
| shape-grouped-current-mix | current | 6 | 5% | 4.25s | 0.1838 | 0.1808 | 1.67% | 100.77s | 1.567 |
| shape-grouped-current-mix | current | 6 | 10% | 2.50s | 0.2345 | 0.2149 | 8.33% | 128.57s | 1.838 |
| shape-grouped-current-mix | slo_no_preemption_lookup | 4 | 5% | 4.25s | 0.2166 | 0.2166 | 0.00% | 27.89s | 1.124 |
| shape-grouped-current-mix | slo_no_preemption_lookup | 4 | 10% | 4.25s | 0.2166 | 0.2166 | 0.00% | 27.89s | 1.124 |
| shape-grouped-current-mix | slo_no_preemption_lookup | 6 | 5% | 4.25s | 0.2168 | 0.2168 | 0.00% | 27.12s | 1.111 |
| shape-grouped-current-mix | slo_no_preemption_lookup | 6 | 10% | 4.25s | 0.2168 | 0.2168 | 0.00% | 27.12s | 1.111 |
