# E2E SLO Throughput Frontier

| Workload | Interarrival | Policy | SLO scale | Repeats | Miss rate | Goodput | Throughput | P95 latency | Mean bucket | P5 TTD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current-mix | 2.50s | current | 4 | 1 | 45.00% | 0.1067 | 0.1941 | 156.36s | 1.257 | -75475.9ms |
| current-mix | 2.50s | current | 6 | 1 | 13.33% | 0.1748 | 0.2016 | 141.62s | 1.303 | -29301.2ms |
| current-mix | 2.50s | slo_no_preemption_guarded | 4 | 1 | 33.33% | 0.1362 | 0.2043 | 147.60s | 1.589 | -62421.1ms |
| current-mix | 2.50s | slo_no_preemption_guarded | 6 | 1 | 26.67% | 0.1592 | 0.2171 | 134.07s | 1.751 | -24025.7ms |
| current-mix | 2.50s | slo_no_preemption_lookup | 4 | 1 | 36.67% | 0.1009 | 0.1594 | 243.97s | 1.658 | -158100.2ms |
| current-mix | 2.50s | slo_no_preemption_lookup | 6 | 1 | 21.67% | 0.1518 | 0.1937 | 180.73s | 1.817 | -70700.1ms |
| current-mix | 4.25s | current | 4 | 1 | 31.67% | 0.1179 | 0.1726 | 102.07s | 1.231 | -23199.0ms |
| current-mix | 4.25s | current | 6 | 1 | 13.33% | 0.1273 | 0.1468 | 147.07s | 1.208 | -30339.9ms |
| current-mix | 4.25s | slo_no_preemption_guarded | 4 | 1 | 1.67% | 0.2010 | 0.2044 | 60.17s | 1.135 | 24352.7ms |
| current-mix | 4.25s | slo_no_preemption_guarded | 6 | 1 | 0.00% | 0.2076 | 0.2076 | 60.62s | 1.107 | 53236.5ms |
| current-mix | 4.25s | slo_no_preemption_lookup | 4 | 1 | 6.67% | 0.1906 | 0.2042 | 82.43s | 1.198 | -103.5ms |
| current-mix | 4.25s | slo_no_preemption_lookup | 6 | 1 | 0.00% | 0.1949 | 0.1949 | 65.67s | 1.115 | 50887.7ms |
| shape-grouped-current-mix | 2.50s | current | 4 | 1 | 31.67% | 0.1700 | 0.2488 | 97.75s | 1.829 | -21987.2ms |
| shape-grouped-current-mix | 2.50s | current | 6 | 1 | 8.33% | 0.2149 | 0.2345 | 128.57s | 1.838 | -18825.1ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_guarded | 4 | 1 | 23.33% | 0.1983 | 0.2587 | 116.66s | 1.923 | -19728.9ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_guarded | 6 | 1 | 8.33% | 0.2212 | 0.2413 | 112.05s | 1.736 | -94.1ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_lookup | 4 | 1 | 35.00% | 0.1418 | 0.2181 | 145.14s | 1.993 | -65633.1ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_lookup | 6 | 1 | 13.33% | 0.2020 | 0.2330 | 130.74s | 2.042 | -23928.1ms |
| shape-grouped-current-mix | 4.25s | current | 4 | 1 | 0.00% | 0.2083 | 0.2083 | 57.08s | 1.572 | 23337.3ms |
| shape-grouped-current-mix | 4.25s | current | 6 | 1 | 1.67% | 0.1808 | 0.1838 | 100.77s | 1.567 | 18392.3ms |
| shape-grouped-current-mix | 4.25s | slo_no_preemption_guarded | 4 | 1 | 0.00% | 0.2169 | 0.2169 | 26.77s | 1.091 | 47295.4ms |
| shape-grouped-current-mix | 4.25s | slo_no_preemption_guarded | 6 | 1 | 0.00% | 0.2167 | 0.2167 | 26.78s | 1.096 | 80589.3ms |
| shape-grouped-current-mix | 4.25s | slo_no_preemption_lookup | 4 | 1 | 0.00% | 0.2166 | 0.2166 | 27.89s | 1.124 | 47173.2ms |
| shape-grouped-current-mix | 4.25s | slo_no_preemption_lookup | 6 | 1 | 0.00% | 0.2168 | 0.2168 | 27.12s | 1.111 | 80189.2ms |

## Current vs slo_no_preemption_guarded

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|
| current-mix | 2.50s | 4 | 11.67 pp | 27.63% | 5.29% | -5.61% | +0.332 |
| current-mix | 2.50s | 6 | -13.33 pp | -8.89% | 7.68% | -5.33% | +0.448 |
| current-mix | 4.25s | 4 | 30.00 pp | 70.48% | 18.47% | -41.05% | -0.096 |
| current-mix | 4.25s | 6 | 13.33 pp | 63.15% | 41.39% | -58.78% | -0.101 |
| shape-grouped-current-mix | 2.50s | 4 | 8.33 pp | 16.63% | 3.96% | 19.35% | +0.094 |
| shape-grouped-current-mix | 2.50s | 6 | 0.00 pp | 2.91% | 2.91% | -12.85% | -0.102 |
| shape-grouped-current-mix | 4.25s | 4 | 0.00 pp | 4.08% | 4.08% | -53.10% | -0.481 |
| shape-grouped-current-mix | 4.25s | 6 | 1.67 pp | 19.86% | 17.86% | -73.43% | -0.472 |

## Frontier by Miss Threshold

| Workload | Policy | SLO scale | Miss threshold | Selected interarrival | Throughput | Goodput | Miss rate | P95 latency | Mean bucket |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current-mix | current | 4 | 5% | none | | | | | |
| current-mix | current | 4 | 10% | none | | | | | |
| current-mix | current | 6 | 5% | none | | | | | |
| current-mix | current | 6 | 10% | none | | | | | |
| current-mix | slo_no_preemption_guarded | 4 | 5% | 4.25s | 0.2044 | 0.2010 | 1.67% | 60.17s | 1.135 |
| current-mix | slo_no_preemption_guarded | 4 | 10% | 4.25s | 0.2044 | 0.2010 | 1.67% | 60.17s | 1.135 |
| current-mix | slo_no_preemption_guarded | 6 | 5% | 4.25s | 0.2076 | 0.2076 | 0.00% | 60.62s | 1.107 |
| current-mix | slo_no_preemption_guarded | 6 | 10% | 4.25s | 0.2076 | 0.2076 | 0.00% | 60.62s | 1.107 |
| current-mix | slo_no_preemption_lookup | 4 | 5% | none | | | | | |
| current-mix | slo_no_preemption_lookup | 4 | 10% | 4.25s | 0.2042 | 0.1906 | 6.67% | 82.43s | 1.198 |
| current-mix | slo_no_preemption_lookup | 6 | 5% | 4.25s | 0.1949 | 0.1949 | 0.00% | 65.67s | 1.115 |
| current-mix | slo_no_preemption_lookup | 6 | 10% | 4.25s | 0.1949 | 0.1949 | 0.00% | 65.67s | 1.115 |
| shape-grouped-current-mix | current | 4 | 5% | 4.25s | 0.2083 | 0.2083 | 0.00% | 57.08s | 1.572 |
| shape-grouped-current-mix | current | 4 | 10% | 4.25s | 0.2083 | 0.2083 | 0.00% | 57.08s | 1.572 |
| shape-grouped-current-mix | current | 6 | 5% | 4.25s | 0.1838 | 0.1808 | 1.67% | 100.77s | 1.567 |
| shape-grouped-current-mix | current | 6 | 10% | 2.50s | 0.2345 | 0.2149 | 8.33% | 128.57s | 1.838 |
| shape-grouped-current-mix | slo_no_preemption_guarded | 4 | 5% | 4.25s | 0.2169 | 0.2169 | 0.00% | 26.77s | 1.091 |
| shape-grouped-current-mix | slo_no_preemption_guarded | 4 | 10% | 4.25s | 0.2169 | 0.2169 | 0.00% | 26.77s | 1.091 |
| shape-grouped-current-mix | slo_no_preemption_guarded | 6 | 5% | 4.25s | 0.2167 | 0.2167 | 0.00% | 26.78s | 1.096 |
| shape-grouped-current-mix | slo_no_preemption_guarded | 6 | 10% | 2.50s | 0.2413 | 0.2212 | 8.33% | 112.05s | 1.736 |
| shape-grouped-current-mix | slo_no_preemption_lookup | 4 | 5% | 4.25s | 0.2166 | 0.2166 | 0.00% | 27.89s | 1.124 |
| shape-grouped-current-mix | slo_no_preemption_lookup | 4 | 10% | 4.25s | 0.2166 | 0.2166 | 0.00% | 27.89s | 1.124 |
| shape-grouped-current-mix | slo_no_preemption_lookup | 6 | 5% | 4.25s | 0.2168 | 0.2168 | 0.00% | 27.12s | 1.111 |
| shape-grouped-current-mix | slo_no_preemption_lookup | 6 | 10% | 4.25s | 0.2168 | 0.2168 | 0.00% | 27.12s | 1.111 |
