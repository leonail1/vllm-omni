# E2E SLO Throughput Frontier

| Workload | Interarrival | Policy | SLO scale | Repeats | Miss rate | Goodput | Throughput | P95 latency | Mean bucket | P5 TTD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bursty | 2.50s | current | 4 | 1 | 40.00% | 0.1125 | 0.1875 | 163.65s | 1.247 | -85601.0ms |
| bursty | 2.50s | current | 6 | 1 | 36.67% | 0.0940 | 0.1484 | 284.21s | 1.131 | -170833.6ms |
| bursty | 2.50s | slo_no_preemption_adaptive_guarded | 4 | 1 | 36.67% | 0.1234 | 0.1948 | 162.22s | 1.715 | -84312.3ms |
| bursty | 2.50s | slo_no_preemption_adaptive_guarded | 6 | 1 | 8.33% | 0.2389 | 0.2607 | 118.83s | 1.900 | -4491.1ms |
| bursty | 2.50s | slo_no_preemption_guarded | 4 | 1 | 40.00% | 0.1278 | 0.2130 | 180.32s | 1.678 | -94197.8ms |
| bursty | 2.50s | slo_no_preemption_guarded | 6 | 1 | 20.00% | 0.1695 | 0.2119 | 157.54s | 1.784 | -35591.9ms |
| bursty | 2.50s | slo_no_preemption_lookup | 4 | 1 | 46.67% | 0.1155 | 0.2166 | 161.62s | 1.736 | -85411.1ms |
| bursty | 2.50s | slo_no_preemption_lookup | 6 | 1 | 26.67% | 0.1392 | 0.1898 | 187.18s | 1.823 | -65463.7ms |
| large-heavy | 2.50s | current | 4 | 1 | 63.33% | 0.0494 | 0.1279 | 261.25s | 1.450 | -169455.3ms |
| large-heavy | 2.50s | current | 6 | 1 | 33.33% | 0.1198 | 0.1797 | 179.54s | 1.453 | -44639.7ms |
| large-heavy | 2.50s | slo_no_preemption_adaptive_guarded | 4 | 1 | 45.00% | 0.0832 | 0.1513 | 232.72s | 2.056 | -112346.3ms |
| large-heavy | 2.50s | slo_no_preemption_adaptive_guarded | 6 | 1 | 23.33% | 0.1435 | 0.1872 | 218.11s | 2.011 | -68203.8ms |
| large-heavy | 2.50s | slo_no_preemption_guarded | 4 | 1 | 53.33% | 0.0631 | 0.1240 | 264.67s | 1.700 | -157209.4ms |
| large-heavy | 2.50s | slo_no_preemption_guarded | 6 | 1 | 38.33% | 0.1046 | 0.1697 | 198.39s | 1.662 | -50816.2ms |
| large-heavy | 2.50s | slo_no_preemption_lookup | 4 | 1 | 41.67% | 0.0818 | 0.1308 | 277.54s | 1.731 | -159902.5ms |
| large-heavy | 2.50s | slo_no_preemption_lookup | 6 | 1 | 31.67% | 0.1298 | 0.1899 | 195.34s | 1.900 | -41178.2ms |
| rectangular-mix | 2.50s | current | 4 | 1 | 51.67% | 0.0784 | 0.1623 | 201.75s | 1.017 | -120150.5ms |
| rectangular-mix | 2.50s | current | 6 | 1 | 31.67% | 0.1221 | 0.1787 | 162.79s | 1.034 | -56400.1ms |
| rectangular-mix | 2.50s | slo_no_preemption_adaptive_guarded | 4 | 1 | 38.33% | 0.1033 | 0.1674 | 218.55s | 1.319 | -129307.9ms |
| rectangular-mix | 2.50s | slo_no_preemption_adaptive_guarded | 6 | 1 | 23.33% | 0.1697 | 0.2213 | 158.40s | 1.323 | -61593.4ms |
| rectangular-mix | 2.50s | slo_no_preemption_guarded | 4 | 1 | 41.67% | 0.1035 | 0.1774 | 189.35s | 1.334 | -104836.4ms |
| rectangular-mix | 2.50s | slo_no_preemption_guarded | 6 | 1 | 18.33% | 0.1832 | 0.2243 | 137.13s | 1.395 | -24974.7ms |
| rectangular-mix | 2.50s | slo_no_preemption_lookup | 4 | 1 | 46.67% | 0.1096 | 0.2055 | 180.02s | 1.369 | -101541.4ms |
| rectangular-mix | 2.50s | slo_no_preemption_lookup | 6 | 1 | 21.67% | 0.1565 | 0.1998 | 151.64s | 1.375 | -50187.8ms |

## Current vs slo_no_preemption_adaptive_guarded

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|
| bursty | 2.50s | 4 | 3.33 pp | 9.71% | 3.94% | -0.88% | +0.468 |
| bursty | 2.50s | 6 | 28.33 pp | 154.29% | 75.69% | -58.19% | +0.769 |
| large-heavy | 2.50s | 4 | 18.33 pp | 68.46% | 18.22% | -10.92% | +0.606 |
| large-heavy | 2.50s | 6 | 10.00 pp | 19.81% | 4.18% | 21.48% | +0.557 |
| rectangular-mix | 2.50s | 4 | 13.33 pp | 31.64% | 3.18% | 8.33% | +0.302 |
| rectangular-mix | 2.50s | 6 | 8.33 pp | 38.97% | 23.86% | -2.69% | +0.288 |

## Frontier by Miss Threshold

| Workload | Policy | SLO scale | Miss threshold | Selected interarrival | Throughput | Goodput | Miss rate | P95 latency | Mean bucket |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bursty | current | 4 | 5% | none | | | | | |
| bursty | current | 4 | 10% | none | | | | | |
| bursty | current | 6 | 5% | none | | | | | |
| bursty | current | 6 | 10% | none | | | | | |
| bursty | slo_no_preemption_adaptive_guarded | 4 | 5% | none | | | | | |
| bursty | slo_no_preemption_adaptive_guarded | 4 | 10% | none | | | | | |
| bursty | slo_no_preemption_adaptive_guarded | 6 | 5% | none | | | | | |
| bursty | slo_no_preemption_adaptive_guarded | 6 | 10% | 2.50s | 0.2607 | 0.2389 | 8.33% | 118.83s | 1.900 |
| bursty | slo_no_preemption_guarded | 4 | 5% | none | | | | | |
| bursty | slo_no_preemption_guarded | 4 | 10% | none | | | | | |
| bursty | slo_no_preemption_guarded | 6 | 5% | none | | | | | |
| bursty | slo_no_preemption_guarded | 6 | 10% | none | | | | | |
| bursty | slo_no_preemption_lookup | 4 | 5% | none | | | | | |
| bursty | slo_no_preemption_lookup | 4 | 10% | none | | | | | |
| bursty | slo_no_preemption_lookup | 6 | 5% | none | | | | | |
| bursty | slo_no_preemption_lookup | 6 | 10% | none | | | | | |
| large-heavy | current | 4 | 5% | none | | | | | |
| large-heavy | current | 4 | 10% | none | | | | | |
| large-heavy | current | 6 | 5% | none | | | | | |
| large-heavy | current | 6 | 10% | none | | | | | |
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
| rectangular-mix | current | 4 | 5% | none | | | | | |
| rectangular-mix | current | 4 | 10% | none | | | | | |
| rectangular-mix | current | 6 | 5% | none | | | | | |
| rectangular-mix | current | 6 | 10% | none | | | | | |
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
