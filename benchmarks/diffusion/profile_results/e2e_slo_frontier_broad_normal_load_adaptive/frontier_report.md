# E2E SLO Throughput Frontier

| Workload | Interarrival | Policy | SLO scale | Repeats | Miss rate | Goodput | Throughput | P95 latency | Mean bucket | P5 TTD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bursty | 4.25s | current | 4 | 1 | 10.00% | 0.1652 | 0.1835 | 82.91s | 1.217 | -8344.4ms |
| bursty | 4.25s | current | 6 | 1 | 1.67% | 0.2019 | 0.2053 | 89.07s | 1.248 | 18305.8ms |
| bursty | 4.25s | slo_no_preemption_adaptive_guarded | 4 | 1 | 11.67% | 0.1738 | 0.1967 | 87.27s | 1.312 | 1563.5ms |
| bursty | 4.25s | slo_no_preemption_adaptive_guarded | 6 | 1 | 0.00% | 0.2003 | 0.2003 | 85.78s | 1.371 | 22610.3ms |
| bursty | 4.25s | slo_no_preemption_guarded | 4 | 1 | 23.33% | 0.1371 | 0.1788 | 131.69s | 1.453 | -55208.9ms |
| bursty | 4.25s | slo_no_preemption_guarded | 6 | 1 | 10.00% | 0.1674 | 0.1860 | 132.33s | 1.527 | -22757.6ms |
| bursty | 4.25s | slo_no_preemption_lookup | 4 | 1 | 10.00% | 0.1779 | 0.1977 | 80.68s | 1.453 | 1045.6ms |
| bursty | 4.25s | slo_no_preemption_lookup | 6 | 1 | 3.33% | 0.1857 | 0.1921 | 91.63s | 1.548 | 21710.6ms |
| large-heavy | 4.25s | current | 4 | 1 | 38.33% | 0.1094 | 0.1773 | 117.94s | 1.330 | -21308.5ms |
| large-heavy | 4.25s | current | 6 | 1 | 25.00% | 0.1040 | 0.1386 | 181.78s | 1.215 | -52167.9ms |
| large-heavy | 4.25s | slo_no_preemption_adaptive_guarded | 4 | 1 | 18.33% | 0.1585 | 0.1941 | 96.55s | 1.398 | -19284.5ms |
| large-heavy | 4.25s | slo_no_preemption_adaptive_guarded | 6 | 1 | 5.00% | 0.1841 | 0.1938 | 99.75s | 1.365 | 14498.6ms |
| large-heavy | 4.25s | slo_no_preemption_guarded | 4 | 1 | 18.33% | 0.1417 | 0.1735 | 116.57s | 1.354 | -39519.7ms |
| large-heavy | 4.25s | slo_no_preemption_guarded | 6 | 1 | 10.00% | 0.1523 | 0.1693 | 117.22s | 1.379 | -9168.7ms |
| large-heavy | 4.25s | slo_no_preemption_lookup | 4 | 1 | 20.00% | 0.1375 | 0.1718 | 117.57s | 1.435 | -49984.2ms |
| large-heavy | 4.25s | slo_no_preemption_lookup | 6 | 1 | 5.00% | 0.1771 | 0.1864 | 97.94s | 1.449 | 2534.6ms |
| rectangular-mix | 4.25s | current | 4 | 1 | 30.00% | 0.1115 | 0.1593 | 105.96s | 1.000 | -28196.3ms |
| rectangular-mix | 4.25s | current | 6 | 1 | 10.00% | 0.1572 | 0.1746 | 104.17s | 1.048 | -66.9ms |
| rectangular-mix | 4.25s | slo_no_preemption_adaptive_guarded | 4 | 1 | 0.00% | 0.1993 | 0.1993 | 54.02s | 1.041 | 15004.7ms |
| rectangular-mix | 4.25s | slo_no_preemption_adaptive_guarded | 6 | 1 | 0.00% | 0.1999 | 0.1999 | 56.71s | 1.018 | 46845.5ms |
| rectangular-mix | 4.25s | slo_no_preemption_guarded | 4 | 1 | 0.00% | 0.1974 | 0.1974 | 55.09s | 1.000 | 15361.1ms |
| rectangular-mix | 4.25s | slo_no_preemption_guarded | 6 | 1 | 0.00% | 0.1913 | 0.1913 | 71.99s | 1.000 | 37935.6ms |
| rectangular-mix | 4.25s | slo_no_preemption_lookup | 4 | 1 | 0.00% | 0.2017 | 0.2017 | 53.94s | 1.023 | 16467.1ms |
| rectangular-mix | 4.25s | slo_no_preemption_lookup | 6 | 1 | 0.00% | 0.2049 | 0.2049 | 44.28s | 1.021 | 60468.6ms |

## Current vs slo_no_preemption_adaptive_guarded

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|
| bursty | 4.25s | 4 | -1.67 pp | 5.22% | 7.20% | 5.26% | +0.095 |
| bursty | 4.25s | 6 | 1.67 pp | -0.80% | -2.45% | -3.69% | +0.123 |
| large-heavy | 4.25s | 4 | 20.00 pp | 44.98% | 9.47% | -18.14% | +0.068 |
| large-heavy | 4.25s | 6 | 20.00 pp | 77.13% | 39.84% | -45.13% | +0.150 |
| rectangular-mix | 4.25s | 4 | 30.00 pp | 78.81% | 25.17% | -49.02% | +0.041 |
| rectangular-mix | 4.25s | 6 | 10.00 pp | 27.20% | 14.48% | -45.56% | -0.030 |

## Frontier by Miss Threshold

| Workload | Policy | SLO scale | Miss threshold | Selected interarrival | Throughput | Goodput | Miss rate | P95 latency | Mean bucket |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bursty | current | 4 | 5% | none | | | | | |
| bursty | current | 4 | 10% | 4.25s | 0.1835 | 0.1652 | 10.00% | 82.91s | 1.217 |
| bursty | current | 6 | 5% | 4.25s | 0.2053 | 0.2019 | 1.67% | 89.07s | 1.248 |
| bursty | current | 6 | 10% | 4.25s | 0.2053 | 0.2019 | 1.67% | 89.07s | 1.248 |
| bursty | slo_no_preemption_adaptive_guarded | 4 | 5% | none | | | | | |
| bursty | slo_no_preemption_adaptive_guarded | 4 | 10% | none | | | | | |
| bursty | slo_no_preemption_adaptive_guarded | 6 | 5% | 4.25s | 0.2003 | 0.2003 | 0.00% | 85.78s | 1.371 |
| bursty | slo_no_preemption_adaptive_guarded | 6 | 10% | 4.25s | 0.2003 | 0.2003 | 0.00% | 85.78s | 1.371 |
| bursty | slo_no_preemption_guarded | 4 | 5% | none | | | | | |
| bursty | slo_no_preemption_guarded | 4 | 10% | none | | | | | |
| bursty | slo_no_preemption_guarded | 6 | 5% | none | | | | | |
| bursty | slo_no_preemption_guarded | 6 | 10% | 4.25s | 0.1860 | 0.1674 | 10.00% | 132.33s | 1.527 |
| bursty | slo_no_preemption_lookup | 4 | 5% | none | | | | | |
| bursty | slo_no_preemption_lookup | 4 | 10% | 4.25s | 0.1977 | 0.1779 | 10.00% | 80.68s | 1.453 |
| bursty | slo_no_preemption_lookup | 6 | 5% | 4.25s | 0.1921 | 0.1857 | 3.33% | 91.63s | 1.548 |
| bursty | slo_no_preemption_lookup | 6 | 10% | 4.25s | 0.1921 | 0.1857 | 3.33% | 91.63s | 1.548 |
| large-heavy | current | 4 | 5% | none | | | | | |
| large-heavy | current | 4 | 10% | none | | | | | |
| large-heavy | current | 6 | 5% | none | | | | | |
| large-heavy | current | 6 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_adaptive_guarded | 4 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_adaptive_guarded | 4 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_adaptive_guarded | 6 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_adaptive_guarded | 6 | 10% | 4.25s | 0.1938 | 0.1841 | 5.00% | 99.75s | 1.365 |
| large-heavy | slo_no_preemption_guarded | 4 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_guarded | 4 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_guarded | 6 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_guarded | 6 | 10% | 4.25s | 0.1693 | 0.1523 | 10.00% | 117.22s | 1.379 |
| large-heavy | slo_no_preemption_lookup | 4 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_lookup | 4 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_lookup | 6 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_lookup | 6 | 10% | 4.25s | 0.1864 | 0.1771 | 5.00% | 97.94s | 1.449 |
| rectangular-mix | current | 4 | 5% | none | | | | | |
| rectangular-mix | current | 4 | 10% | none | | | | | |
| rectangular-mix | current | 6 | 5% | none | | | | | |
| rectangular-mix | current | 6 | 10% | 4.25s | 0.1746 | 0.1572 | 10.00% | 104.17s | 1.048 |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 4 | 5% | 4.25s | 0.1993 | 0.1993 | 0.00% | 54.02s | 1.041 |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 4 | 10% | 4.25s | 0.1993 | 0.1993 | 0.00% | 54.02s | 1.041 |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 6 | 5% | 4.25s | 0.1999 | 0.1999 | 0.00% | 56.71s | 1.018 |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 6 | 10% | 4.25s | 0.1999 | 0.1999 | 0.00% | 56.71s | 1.018 |
| rectangular-mix | slo_no_preemption_guarded | 4 | 5% | 4.25s | 0.1974 | 0.1974 | 0.00% | 55.09s | 1.000 |
| rectangular-mix | slo_no_preemption_guarded | 4 | 10% | 4.25s | 0.1974 | 0.1974 | 0.00% | 55.09s | 1.000 |
| rectangular-mix | slo_no_preemption_guarded | 6 | 5% | 4.25s | 0.1913 | 0.1913 | 0.00% | 71.99s | 1.000 |
| rectangular-mix | slo_no_preemption_guarded | 6 | 10% | 4.25s | 0.1913 | 0.1913 | 0.00% | 71.99s | 1.000 |
| rectangular-mix | slo_no_preemption_lookup | 4 | 5% | 4.25s | 0.2017 | 0.2017 | 0.00% | 53.94s | 1.023 |
| rectangular-mix | slo_no_preemption_lookup | 4 | 10% | 4.25s | 0.2017 | 0.2017 | 0.00% | 53.94s | 1.023 |
| rectangular-mix | slo_no_preemption_lookup | 6 | 5% | 4.25s | 0.2049 | 0.2049 | 0.00% | 44.28s | 1.021 |
| rectangular-mix | slo_no_preemption_lookup | 6 | 10% | 4.25s | 0.2049 | 0.2049 | 0.00% | 44.28s | 1.021 |
