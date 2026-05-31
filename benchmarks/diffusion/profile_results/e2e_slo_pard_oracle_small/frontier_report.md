# E2E SLO Throughput Frontier

| Workload | Interarrival | Policy | SLO scale | Repeats | Miss rate | Goodput | Throughput | P95 latency | Mean bucket | P5 TTD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current-mix | 2.50s | current | 4 | 1 | 38.33% | 0.1211 | 0.1964 | 160.67s | 1.170 | -84451.1ms |
| current-mix | 2.50s | current | 6 | 1 | 16.67% | 0.1398 | 0.1677 | 177.60s | 1.341 | -52605.9ms |
| current-mix | 2.50s | slo_no_preemption_adaptive_guarded | 4 | 1 | 33.33% | 0.1692 | 0.2537 | 113.27s | 1.799 | -32990.4ms |
| current-mix | 2.50s | slo_no_preemption_adaptive_guarded | 6 | 1 | 11.67% | 0.1901 | 0.2152 | 135.97s | 1.796 | -26621.5ms |
| current-mix | 2.50s | slo_no_preemption_guarded | 4 | 1 | 35.00% | 0.1349 | 0.2075 | 171.84s | 1.572 | -85495.4ms |
| current-mix | 2.50s | slo_no_preemption_guarded | 6 | 1 | 28.33% | 0.1358 | 0.1895 | 150.22s | 1.735 | -31271.0ms |
| current-mix | 2.50s | slo_no_preemption_pard_dry_run | 4 | 1 | 26.67% | 0.1806 | 0.2463 | 114.27s | 1.546 | -24417.7ms |
| current-mix | 2.50s | slo_no_preemption_pard_dry_run | 6 | 1 | 11.67% | 0.2088 | 0.2364 | 148.98s | 1.735 | -31493.4ms |
| shape-grouped-current-mix | 2.50s | current | 4 | 1 | 35.00% | 0.1410 | 0.2169 | 115.12s | 1.996 | -22941.7ms |
| shape-grouped-current-mix | 2.50s | current | 6 | 1 | 1.67% | 0.2315 | 0.2354 | 97.22s | 1.944 | 21749.8ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_adaptive_guarded | 4 | 1 | 18.33% | 0.2079 | 0.2546 | 98.69s | 1.835 | -18061.9ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_adaptive_guarded | 6 | 1 | 1.67% | 0.2423 | 0.2464 | 96.23s | 1.899 | 17249.6ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_guarded | 4 | 1 | 26.67% | 0.1533 | 0.2090 | 125.22s | 1.820 | -31089.3ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_guarded | 6 | 1 | 3.33% | 0.2446 | 0.2531 | 109.33s | 1.866 | 17283.7ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_pard_dry_run | 4 | 1 | 15.00% | 0.2114 | 0.2487 | 93.69s | 1.910 | -16078.5ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_pard_dry_run | 6 | 1 | 6.67% | 0.2254 | 0.2415 | 104.67s | 1.948 | 6608.8ms |

## Current vs slo_no_preemption_pard_dry_run

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|
| current-mix | 2.50s | 4 | 11.67 pp | 49.19% | 25.46% | -28.88% | +0.376 |
| current-mix | 2.50s | 6 | 5.00 pp | 49.39% | 40.94% | -16.12% | +0.394 |
| shape-grouped-current-mix | 2.50s | 4 | 20.00 pp | 49.88% | 14.62% | -18.61% | -0.086 |
| shape-grouped-current-mix | 2.50s | 6 | -5.00 pp | -2.63% | 2.59% | 7.66% | +0.004 |

## Frontier by Miss Threshold

| Workload | Policy | SLO scale | Miss threshold | Selected interarrival | Throughput | Goodput | Miss rate | P95 latency | Mean bucket |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current-mix | current | 4 | 5% | none | | | | | |
| current-mix | current | 4 | 10% | none | | | | | |
| current-mix | current | 6 | 5% | none | | | | | |
| current-mix | current | 6 | 10% | none | | | | | |
| current-mix | slo_no_preemption_adaptive_guarded | 4 | 5% | none | | | | | |
| current-mix | slo_no_preemption_adaptive_guarded | 4 | 10% | none | | | | | |
| current-mix | slo_no_preemption_adaptive_guarded | 6 | 5% | none | | | | | |
| current-mix | slo_no_preemption_adaptive_guarded | 6 | 10% | none | | | | | |
| current-mix | slo_no_preemption_guarded | 4 | 5% | none | | | | | |
| current-mix | slo_no_preemption_guarded | 4 | 10% | none | | | | | |
| current-mix | slo_no_preemption_guarded | 6 | 5% | none | | | | | |
| current-mix | slo_no_preemption_guarded | 6 | 10% | none | | | | | |
| current-mix | slo_no_preemption_pard_dry_run | 4 | 5% | none | | | | | |
| current-mix | slo_no_preemption_pard_dry_run | 4 | 10% | none | | | | | |
| current-mix | slo_no_preemption_pard_dry_run | 6 | 5% | none | | | | | |
| current-mix | slo_no_preemption_pard_dry_run | 6 | 10% | none | | | | | |
| shape-grouped-current-mix | current | 4 | 5% | none | | | | | |
| shape-grouped-current-mix | current | 4 | 10% | none | | | | | |
| shape-grouped-current-mix | current | 6 | 5% | 2.50s | 0.2354 | 0.2315 | 1.67% | 97.22s | 1.944 |
| shape-grouped-current-mix | current | 6 | 10% | 2.50s | 0.2354 | 0.2315 | 1.67% | 97.22s | 1.944 |
| shape-grouped-current-mix | slo_no_preemption_adaptive_guarded | 4 | 5% | none | | | | | |
| shape-grouped-current-mix | slo_no_preemption_adaptive_guarded | 4 | 10% | none | | | | | |
| shape-grouped-current-mix | slo_no_preemption_adaptive_guarded | 6 | 5% | 2.50s | 0.2464 | 0.2423 | 1.67% | 96.23s | 1.899 |
| shape-grouped-current-mix | slo_no_preemption_adaptive_guarded | 6 | 10% | 2.50s | 0.2464 | 0.2423 | 1.67% | 96.23s | 1.899 |
| shape-grouped-current-mix | slo_no_preemption_guarded | 4 | 5% | none | | | | | |
| shape-grouped-current-mix | slo_no_preemption_guarded | 4 | 10% | none | | | | | |
| shape-grouped-current-mix | slo_no_preemption_guarded | 6 | 5% | 2.50s | 0.2531 | 0.2446 | 3.33% | 109.33s | 1.866 |
| shape-grouped-current-mix | slo_no_preemption_guarded | 6 | 10% | 2.50s | 0.2531 | 0.2446 | 3.33% | 109.33s | 1.866 |
| shape-grouped-current-mix | slo_no_preemption_pard_dry_run | 4 | 5% | none | | | | | |
| shape-grouped-current-mix | slo_no_preemption_pard_dry_run | 4 | 10% | none | | | | | |
| shape-grouped-current-mix | slo_no_preemption_pard_dry_run | 6 | 5% | none | | | | | |
| shape-grouped-current-mix | slo_no_preemption_pard_dry_run | 6 | 10% | 2.50s | 0.2415 | 0.2254 | 6.67% | 104.67s | 1.948 |
