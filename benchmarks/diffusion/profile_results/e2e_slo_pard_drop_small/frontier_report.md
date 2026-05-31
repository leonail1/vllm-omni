# E2E SLO Throughput Frontier

| Workload | Interarrival | Policy | SLO scale | Repeats | Miss rate | Goodput | Throughput | P95 latency | Mean bucket | P5 TTD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current-mix | 2.50s | current | 4 | 1 | 38.33% | 0.1211 | 0.1964 | 160.67s | 1.170 | -84451.1ms |
| current-mix | 2.50s | current | 6 | 1 | 16.67% | 0.1398 | 0.1677 | 177.60s | 1.341 | -52605.9ms |
| current-mix | 2.50s | slo_no_preemption_adaptive_guarded | 4 | 1 | 33.33% | 0.1692 | 0.2537 | 113.27s | 1.799 | -32990.4ms |
| current-mix | 2.50s | slo_no_preemption_adaptive_guarded | 6 | 1 | 11.67% | 0.1901 | 0.2152 | 135.97s | 1.796 | -26621.5ms |
| current-mix | 2.50s | slo_no_preemption_guarded | 4 | 1 | 35.00% | 0.1349 | 0.2075 | 171.84s | 1.572 | -85495.4ms |
| current-mix | 2.50s | slo_no_preemption_guarded | 6 | 1 | 28.33% | 0.1358 | 0.1895 | 150.22s | 1.735 | -31271.0ms |
| current-mix | 2.50s | slo_no_preemption_pard_admission_drop | 4 | 1 | 36.67% | 0.1962 | 0.1962 | 56.15s | 1.327 | 18384.9ms |
| current-mix | 2.50s | slo_no_preemption_pard_admission_drop | 6 | 1 | 63.33% | 0.1265 | 0.1265 | 44.47s | 1.300 | 63211.0ms |
| current-mix | 2.50s | slo_no_preemption_pard_dry_run | 4 | 1 | 26.67% | 0.1806 | 0.2463 | 114.27s | 1.546 | -24417.7ms |
| current-mix | 2.50s | slo_no_preemption_pard_dry_run | 6 | 1 | 11.67% | 0.2088 | 0.2364 | 148.98s | 1.735 | -31493.4ms |
| current-mix | 2.50s | slo_no_preemption_pard_step_drop | 4 | 1 | 48.33% | 0.1493 | 0.1493 | 33.91s | 1.175 | 43149.5ms |
| current-mix | 2.50s | slo_no_preemption_pard_step_drop | 6 | 1 | 40.00% | 0.1685 | 0.1685 | 60.38s | 1.398 | 57421.7ms |
| shape-grouped-current-mix | 2.50s | current | 4 | 1 | 35.00% | 0.1410 | 0.2169 | 115.12s | 1.996 | -22941.7ms |
| shape-grouped-current-mix | 2.50s | current | 6 | 1 | 1.67% | 0.2315 | 0.2354 | 97.22s | 1.944 | 21749.8ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_adaptive_guarded | 4 | 1 | 18.33% | 0.2079 | 0.2546 | 98.69s | 1.835 | -18061.9ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_adaptive_guarded | 6 | 1 | 1.67% | 0.2423 | 0.2464 | 96.23s | 1.899 | 17249.6ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_guarded | 4 | 1 | 26.67% | 0.1533 | 0.2090 | 125.22s | 1.820 | -31089.3ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_guarded | 6 | 1 | 3.33% | 0.2446 | 0.2531 | 109.33s | 1.866 | 17283.7ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_pard_admission_drop | 4 | 1 | 38.33% | 0.1703 | 0.1703 | 71.80s | 1.602 | 28571.3ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_pard_admission_drop | 6 | 1 | 20.00% | 0.2390 | 0.2390 | 88.26s | 1.990 | 34809.1ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_pard_dry_run | 4 | 1 | 15.00% | 0.2114 | 0.2487 | 93.69s | 1.910 | -16078.5ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_pard_dry_run | 6 | 1 | 6.67% | 0.2254 | 0.2415 | 104.67s | 1.948 | 6608.8ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_pard_step_drop | 4 | 1 | 40.00% | 0.1632 | 0.1632 | 73.40s | 1.785 | 23077.1ms |
| shape-grouped-current-mix | 2.50s | slo_no_preemption_pard_step_drop | 6 | 1 | 28.33% | 0.2327 | 0.2327 | 78.08s | 1.857 | 56900.3ms |

## Current vs slo_no_preemption_pard_step_drop

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|
| current-mix | 2.50s | 4 | -10.00 pp | 23.28% | -23.98% | -78.89% | +0.005 |
| current-mix | 2.50s | 6 | -23.33 pp | 20.57% | 0.48% | -66.00% | +0.057 |
| shape-grouped-current-mix | 2.50s | 4 | -5.00 pp | 15.73% | -24.78% | -36.24% | -0.211 |
| shape-grouped-current-mix | 2.50s | 6 | -26.67 pp | 0.50% | -1.17% | -19.69% | -0.087 |

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
| current-mix | slo_no_preemption_pard_admission_drop | 4 | 5% | none | | | | | |
| current-mix | slo_no_preemption_pard_admission_drop | 4 | 10% | none | | | | | |
| current-mix | slo_no_preemption_pard_admission_drop | 6 | 5% | none | | | | | |
| current-mix | slo_no_preemption_pard_admission_drop | 6 | 10% | none | | | | | |
| current-mix | slo_no_preemption_pard_dry_run | 4 | 5% | none | | | | | |
| current-mix | slo_no_preemption_pard_dry_run | 4 | 10% | none | | | | | |
| current-mix | slo_no_preemption_pard_dry_run | 6 | 5% | none | | | | | |
| current-mix | slo_no_preemption_pard_dry_run | 6 | 10% | none | | | | | |
| current-mix | slo_no_preemption_pard_step_drop | 4 | 5% | none | | | | | |
| current-mix | slo_no_preemption_pard_step_drop | 4 | 10% | none | | | | | |
| current-mix | slo_no_preemption_pard_step_drop | 6 | 5% | none | | | | | |
| current-mix | slo_no_preemption_pard_step_drop | 6 | 10% | none | | | | | |
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
| shape-grouped-current-mix | slo_no_preemption_pard_admission_drop | 4 | 5% | none | | | | | |
| shape-grouped-current-mix | slo_no_preemption_pard_admission_drop | 4 | 10% | none | | | | | |
| shape-grouped-current-mix | slo_no_preemption_pard_admission_drop | 6 | 5% | none | | | | | |
| shape-grouped-current-mix | slo_no_preemption_pard_admission_drop | 6 | 10% | none | | | | | |
| shape-grouped-current-mix | slo_no_preemption_pard_dry_run | 4 | 5% | none | | | | | |
| shape-grouped-current-mix | slo_no_preemption_pard_dry_run | 4 | 10% | none | | | | | |
| shape-grouped-current-mix | slo_no_preemption_pard_dry_run | 6 | 5% | none | | | | | |
| shape-grouped-current-mix | slo_no_preemption_pard_dry_run | 6 | 10% | 2.50s | 0.2415 | 0.2254 | 6.67% | 104.67s | 1.948 |
| shape-grouped-current-mix | slo_no_preemption_pard_step_drop | 4 | 5% | none | | | | | |
| shape-grouped-current-mix | slo_no_preemption_pard_step_drop | 4 | 10% | none | | | | | |
| shape-grouped-current-mix | slo_no_preemption_pard_step_drop | 6 | 5% | none | | | | | |
| shape-grouped-current-mix | slo_no_preemption_pard_step_drop | 6 | 10% | none | | | | | |
