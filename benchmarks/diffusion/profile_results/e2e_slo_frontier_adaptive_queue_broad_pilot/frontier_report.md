# E2E SLO Throughput Frontier

| Workload | Interarrival | Policy | SLO scale | Repeats | Miss rate | Goodput | Throughput | P95 latency | Mean bucket | P5 TTD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bursty | 2.50s | slo_no_preemption_adaptive_guarded | 4 | 1 | 23.33% | 0.1654 | 0.2158 | 154.00s | 1.673 | -53905.9ms |
| bursty | 2.50s | slo_no_preemption_adaptive_guarded | 6 | 1 | 15.00% | 0.2053 | 0.2415 | 141.12s | 1.744 | -13413.8ms |
| bursty | 2.50s | slo_no_preemption_guarded | 4 | 1 | 41.67% | 0.1216 | 0.2085 | 142.88s | 1.704 | -65159.7ms |
| bursty | 2.50s | slo_no_preemption_guarded | 6 | 1 | 28.33% | 0.1424 | 0.1987 | 141.20s | 1.773 | -29991.0ms |
| large-heavy | 2.50s | slo_no_preemption_adaptive_guarded | 4 | 1 | 41.67% | 0.0940 | 0.1611 | 239.18s | 1.902 | -147964.9ms |
| large-heavy | 2.50s | slo_no_preemption_adaptive_guarded | 6 | 1 | 31.67% | 0.1099 | 0.1608 | 237.49s | 1.997 | -96130.0ms |
| large-heavy | 2.50s | slo_no_preemption_guarded | 4 | 1 | 45.00% | 0.0957 | 0.1740 | 225.80s | 1.828 | -143164.5ms |
| large-heavy | 2.50s | slo_no_preemption_guarded | 6 | 1 | 38.33% | 0.0950 | 0.1541 | 244.82s | 1.722 | -114011.3ms |
| rectangular-mix | 2.50s | slo_no_preemption_adaptive_guarded | 4 | 1 | 35.00% | 0.1445 | 0.2223 | 122.46s | 1.372 | -48723.0ms |
| rectangular-mix | 2.50s | slo_no_preemption_adaptive_guarded | 6 | 1 | 16.67% | 0.1838 | 0.2205 | 139.47s | 1.369 | -35855.6ms |
| rectangular-mix | 2.50s | slo_no_preemption_guarded | 4 | 1 | 46.67% | 0.1008 | 0.1891 | 173.30s | 1.324 | -93855.7ms |
| rectangular-mix | 2.50s | slo_no_preemption_guarded | 6 | 1 | 33.33% | 0.1111 | 0.1666 | 195.33s | 1.337 | -90019.5ms |

## Current vs slo_no_preemption_adaptive_guarded

| Workload | Interarrival | SLO scale | Miss delta | Goodput change | Throughput change | P95 latency change | Bucket change |
|---|---:|---:|---:|---:|---:|---:|---:|

## Frontier by Miss Threshold

| Workload | Policy | SLO scale | Miss threshold | Selected interarrival | Throughput | Goodput | Miss rate | P95 latency | Mean bucket |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bursty | slo_no_preemption_adaptive_guarded | 4 | 5% | none | | | | | |
| bursty | slo_no_preemption_adaptive_guarded | 4 | 10% | none | | | | | |
| bursty | slo_no_preemption_adaptive_guarded | 6 | 5% | none | | | | | |
| bursty | slo_no_preemption_adaptive_guarded | 6 | 10% | none | | | | | |
| bursty | slo_no_preemption_guarded | 4 | 5% | none | | | | | |
| bursty | slo_no_preemption_guarded | 4 | 10% | none | | | | | |
| bursty | slo_no_preemption_guarded | 6 | 5% | none | | | | | |
| bursty | slo_no_preemption_guarded | 6 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_adaptive_guarded | 4 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_adaptive_guarded | 4 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_adaptive_guarded | 6 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_adaptive_guarded | 6 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_guarded | 4 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_guarded | 4 | 10% | none | | | | | |
| large-heavy | slo_no_preemption_guarded | 6 | 5% | none | | | | | |
| large-heavy | slo_no_preemption_guarded | 6 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 4 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 4 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 6 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_adaptive_guarded | 6 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_guarded | 4 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_guarded | 4 | 10% | none | | | | | |
| rectangular-mix | slo_no_preemption_guarded | 6 | 5% | none | | | | | |
| rectangular-mix | slo_no_preemption_guarded | 6 | 10% | none | | | | | |
