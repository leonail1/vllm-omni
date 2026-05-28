# E2E SLO Throughput Frontier

| Workload | Interarrival | Policy | SLO scale | Repeats | Miss rate | Goodput | Throughput | P95 latency | Mean bucket | P5 TTD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bursty | 2.50s | current | 3 | 1 | 68.33% | 0.0516 | 0.1630 | 201.44s | 1.358 | -132535.0ms |
| bursty | 2.50s | current | 3.5 | 1 | 48.33% | 0.0944 | 0.1827 | 157.38s | 1.485 | -76310.8ms |
| bursty | 2.50s | slo_no_preemption_adaptive_guarded | 3 | 1 | 38.33% | 0.1451 | 0.2353 | 132.62s | 1.739 | -67648.5ms |
| bursty | 2.50s | slo_no_preemption_adaptive_guarded | 3.5 | 1 | 38.33% | 0.1545 | 0.2505 | 117.45s | 1.583 | -37158.5ms |
| bursty | 2.50s | slo_no_preemption_guarded | 3 | 1 | 48.33% | 0.1186 | 0.2295 | 150.35s | 1.717 | -86741.3ms |
| bursty | 2.50s | slo_no_preemption_guarded | 3.5 | 1 | 50.00% | 0.1094 | 0.2187 | 192.63s | 1.609 | -121180.1ms |
| bursty | 2.50s | slo_no_preemption_lookup | 3 | 1 | 53.33% | 0.0944 | 0.2022 | 149.03s | 1.813 | -80256.4ms |
| bursty | 2.50s | slo_no_preemption_lookup | 3.5 | 1 | 53.33% | 0.0990 | 0.2121 | 175.73s | 1.715 | -105215.8ms |
| large-heavy | 2.50s | current | 3 | 1 | 83.33% | 0.0272 | 0.1633 | 202.89s | 1.526 | -140458.6ms |
| large-heavy | 2.50s | current | 3.5 | 1 | 76.67% | 0.0378 | 0.1619 | 197.18s | 1.479 | -116975.8ms |
| large-heavy | 2.50s | slo_no_preemption_adaptive_guarded | 3 | 1 | 41.67% | 0.0781 | 0.1317 | 266.99s | 2.133 | -199644.2ms |
| large-heavy | 2.50s | slo_no_preemption_adaptive_guarded | 3.5 | 1 | 45.00% | 0.1061 | 0.1929 | 198.54s | 2.009 | -110762.4ms |
| large-heavy | 2.50s | slo_no_preemption_guarded | 3 | 1 | 51.67% | 0.0721 | 0.1491 | 239.50s | 1.741 | -157206.1ms |
| large-heavy | 2.50s | slo_no_preemption_guarded | 3.5 | 1 | 46.67% | 0.0784 | 0.1470 | 256.22s | 1.717 | -156938.4ms |
| large-heavy | 2.50s | slo_no_preemption_lookup | 3 | 1 | 60.00% | 0.0763 | 0.1908 | 224.30s | 1.873 | -167582.2ms |
| large-heavy | 2.50s | slo_no_preemption_lookup | 3.5 | 1 | 63.33% | 0.0628 | 0.1712 | 249.19s | 2.001 | -183737.6ms |
| rectangular-mix | 2.50s | current | 3 | 1 | 66.67% | 0.0571 | 0.1713 | 197.60s | 1.034 | -140004.4ms |
| rectangular-mix | 2.50s | current | 3.5 | 1 | 66.67% | 0.0619 | 0.1857 | 159.05s | 1.049 | -88498.6ms |
| rectangular-mix | 2.50s | slo_no_preemption_adaptive_guarded | 3 | 1 | 53.33% | 0.1082 | 0.2318 | 124.46s | 1.353 | -58660.6ms |
| rectangular-mix | 2.50s | slo_no_preemption_adaptive_guarded | 3.5 | 1 | 46.67% | 0.1073 | 0.2013 | 121.47s | 1.321 | -56351.0ms |
| rectangular-mix | 2.50s | slo_no_preemption_guarded | 3 | 1 | 48.33% | 0.0746 | 0.1421 | 266.85s | 1.434 | -211030.4ms |
| rectangular-mix | 2.50s | slo_no_preemption_guarded | 3.5 | 1 | 50.00% | 0.1055 | 0.2110 | 133.45s | 1.345 | -58366.9ms |
| rectangular-mix | 2.50s | slo_no_preemption_lookup | 3 | 1 | 55.00% | 0.0908 | 0.2018 | 163.95s | 1.252 | -99767.8ms |
| rectangular-mix | 2.50s | slo_no_preemption_lookup | 3.5 | 1 | 46.67% | 0.0947 | 0.1776 | 210.40s | 1.413 | -139714.2ms |

## Current vs slo_no_preemption_adaptive_guarded

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|
| bursty | 2.50s | 3 | 30.00 pp | 181.11% | 44.35% | -34.17% | +0.381 |
| bursty | 2.50s | 3.5 | 10.00 pp | 63.67% | 37.13% | -25.37% | +0.098 |
| large-heavy | 2.50s | 3 | 41.67 pp | 187.06% | -19.35% | 31.60% | +0.607 |
| large-heavy | 2.50s | 3.5 | 31.67 pp | 180.91% | 19.17% | 0.69% | +0.531 |
| rectangular-mix | 2.50s | 3 | 13.33 pp | 89.47% | 35.33% | -37.01% | +0.318 |
| rectangular-mix | 2.50s | 3.5 | 20.00 pp | 73.46% | 8.41% | -23.63% | +0.272 |

## Frontier by Miss Threshold

| Workload | Policy | SLO scale | Miss threshold | Selected interarrival | Throughput | Goodput | Miss rate | P95 latency | Mean bucket |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bursty | current | 3 | 5% | none | | | | | |
| bursty | current | 3 | 10% | none | | | | | |
| bursty | current | 3.5 | 5% | none | | | | | |
| bursty | current | 3.5 | 10% | none | | | | | |
| bursty | slo_no_preemption_adaptive_guarded | 3 | 5% | none | | | | | |
| bursty | slo_no_preemption_adaptive_guarded | 3 | 10% | none | | | | | |
| bursty | slo_no_preemption_adaptive_guarded | 3.5 | 5% | none | | | | | |
| bursty | slo_no_preemption_adaptive_guarded | 3.5 | 10% | none | | | | | |
| bursty | slo_no_preemption_guarded | 3 | 5% | none | | | | | |
| bursty | slo_no_preemption_guarded | 3 | 10% | none | | | | | |
| bursty | slo_no_preemption_guarded | 3.5 | 5% | none | | | | | |
| bursty | slo_no_preemption_guarded | 3.5 | 10% | none | | | | | |
| bursty | slo_no_preemption_lookup | 3 | 5% | none | | | | | |
| bursty | slo_no_preemption_lookup | 3 | 10% | none | | | | | |
| bursty | slo_no_preemption_lookup | 3.5 | 5% | none | | | | | |
| bursty | slo_no_preemption_lookup | 3.5 | 10% | none | | | | | |
| large-heavy | current | 3 | 5% | none | | | | | |
| large-heavy | current | 3 | 10% | none | | | | | |
| large-heavy | current | 3.5 | 5% | none | | | | | |
| large-heavy | current | 3.5 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_adaptive_guarded | 3 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_adaptive_guarded | 3 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_adaptive_guarded | 3.5 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_adaptive_guarded | 3.5 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_guarded | 3 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_guarded | 3 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_guarded | 3.5 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_guarded | 3.5 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_lookup | 3 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_lookup | 3 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_lookup | 3.5 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_lookup | 3.5 | 10% | none | | | | | |
| rectangular-mix | current | 3 | 5% | none | | | | | |
| rectangular-mix | current | 3 | 10% | none | | | | | |
| rectangular-mix | current | 3.5 | 5% | none | | | | | |
| rectangular-mix | current | 3.5 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 3 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 3 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 3.5 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 3.5 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_guarded | 3 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_guarded | 3 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_guarded | 3.5 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_guarded | 3.5 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_lookup | 3 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_lookup | 3 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_lookup | 3.5 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_lookup | 3.5 | 10% | none | | | | | |
