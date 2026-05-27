# E2E SLO Throughput Frontier

| Workload | Interarrival | Policy | SLO scale | Repeats | Miss rate | Goodput | Throughput | P95 latency | Mean bucket | P5 TTD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bursty | 2.50s | current | 4 | 1 | 48.33% | 0.0942 | 0.1823 | 175.73s | 1.250 | -106751.5ms |
| bursty | 2.50s | current | 6 | 1 | 10.00% | 0.1738 | 0.1931 | 147.10s | 1.307 | -32158.8ms |
| bursty | 2.50s | slo_no_preemption_lookup | 4 | 1 | 26.67% | 0.1524 | 0.2078 | 141.66s | 1.991 | -34509.2ms |
| bursty | 2.50s | slo_no_preemption_lookup | 6 | 1 | 26.67% | 0.1387 | 0.1891 | 189.11s | 1.896 | -66249.2ms |
| large-heavy | 2.50s | current | 4 | 1 | 56.67% | 0.0653 | 0.1506 | 256.59s | 1.357 | -147441.4ms |
| large-heavy | 2.50s | current | 6 | 1 | 41.67% | 0.1056 | 0.1810 | 186.58s | 1.361 | -50917.7ms |
| large-heavy | 2.50s | slo_no_preemption_lookup | 4 | 1 | 46.67% | 0.1016 | 0.1905 | 235.61s | 1.907 | -158643.1ms |
| large-heavy | 2.50s | slo_no_preemption_lookup | 6 | 1 | 36.67% | 0.0877 | 0.1292 | 273.61s | 1.711 | -115172.1ms |
| rectangular-mix | 2.50s | current | 4 | 1 | 50.00% | 0.0865 | 0.1731 | 171.37s | 1.071 | -103134.8ms |
| rectangular-mix | 2.50s | current | 6 | 1 | 33.33% | 0.1190 | 0.1785 | 166.27s | 1.034 | -57274.7ms |
| rectangular-mix | 2.50s | slo_no_preemption_lookup | 4 | 1 | 41.67% | 0.1303 | 0.2233 | 128.05s | 1.493 | -52704.9ms |
| rectangular-mix | 2.50s | slo_no_preemption_lookup | 6 | 1 | 28.33% | 0.1166 | 0.1626 | 219.49s | 1.400 | -96625.9ms |

## Current vs slo_no_preemption_lookup

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|
| bursty | 2.50s | 4 | 21.67 pp | 61.78% | 13.98% | -19.39% | +0.741 |
| bursty | 2.50s | 6 | -16.67 pp | -20.19% | -2.05% | 28.56% | +0.589 |
| large-heavy | 2.50s | 4 | 10.00 pp | 55.60% | 26.43% | -8.18% | +0.550 |
| large-heavy | 2.50s | 6 | 5.00 pp | -16.94% | -28.60% | 46.65% | +0.350 |
| rectangular-mix | 2.50s | 4 | 8.33 pp | 50.54% | 29.03% | -25.28% | +0.421 |
| rectangular-mix | 2.50s | 6 | 5.00 pp | -2.02% | -8.86% | 32.01% | +0.365 |

## Frontier by Miss Threshold

| Workload | Policy | SLO scale | Miss threshold | Selected interarrival | Throughput | Goodput | Miss rate | P95 latency | Mean bucket |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bursty | current | 4 | 5% | none | | | | | |
| bursty | current | 4 | 10% | none | | | | | |
| bursty | current | 6 | 5% | none | | | | | |
| bursty | current | 6 | 10% | 2.50s | 0.1931 | 0.1738 | 10.00% | 147.10s | 1.307 |
| bursty | slo_no_preemption_lookup | 4 | 5% | none | | | | | |
| bursty | slo_no_preemption_lookup | 4 | 10% | none | | | | | |
| bursty | slo_no_preemption_lookup | 6 | 5% | none | | | | | |
| bursty | slo_no_preemption_lookup | 6 | 10% | none | | | | | |
| large-heavy | current | 4 | 5% | none | | | | | |
| large-heavy | current | 4 | 10% | none | | | | | |
| large-heavy | current | 6 | 5% | none | | | | | |
| large-heavy | current | 6 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_lookup | 4 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_lookup | 4 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_lookup | 6 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_lookup | 6 | 10% | none | | | | | |
| rectangular-mix | current | 4 | 5% | none | | | | | |
| rectangular-mix | current | 4 | 10% | none | | | | | |
| rectangular-mix | current | 6 | 5% | none | | | | | |
| rectangular-mix | current | 6 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_lookup | 4 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_lookup | 4 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_lookup | 6 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_lookup | 6 | 10% | none | | | | | |
