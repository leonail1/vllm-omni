# E2E SLO Ablation Summary

| Workload | Policy | SLO scale | Repeats | Miss rate | Goodput | Throughput | P95 latency | Mean bucket | P5 TTD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bursty | current | 2.5 | 1 | 67.50% | 0.0493 | 0.1518 | 180.03s | 1.161 | -121966.5ms |
| bursty | current | 3 | 1 | 30.00% | 0.1245 | 0.1779 | 108.82s | 1.303 | -49612.6ms |
| bursty | current | 3.5 | 1 | 15.00% | 0.1769 | 0.2081 | 80.08s | 1.182 | -9535.5ms |
| bursty | current | 4 | 1 | 35.00% | 0.1118 | 0.1720 | 145.79s | 1.206 | -60028.5ms |
| bursty | slo_no_preemption_lookup | 2.5 | 1 | 45.00% | 0.1116 | 0.2029 | 154.90s | 1.543 | -92880.6ms |
| bursty | slo_no_preemption_lookup | 3 | 1 | 38.75% | 0.1101 | 0.1798 | 126.60s | 1.650 | -60064.4ms |
| bursty | slo_no_preemption_lookup | 3.5 | 1 | 31.25% | 0.1158 | 0.1685 | 175.36s | 1.569 | -102643.8ms |
| bursty | slo_no_preemption_lookup | 4 | 1 | 20.00% | 0.1553 | 0.1942 | 137.95s | 1.625 | -52012.7ms |
| current-mix | current | 2.5 | 1 | 42.50% | 0.0887 | 0.1542 | 137.44s | 1.139 | -91727.4ms |
| current-mix | current | 3 | 1 | 41.25% | 0.0953 | 0.1622 | 129.10s | 1.169 | -70329.4ms |
| current-mix | current | 3.5 | 1 | 28.75% | 0.1057 | 0.1484 | 198.03s | 1.130 | -127393.2ms |
| current-mix | current | 4 | 1 | 15.00% | 0.1565 | 0.1841 | 92.65s | 1.168 | -12347.5ms |
| current-mix | slo_no_preemption_lookup | 2.5 | 1 | 22.50% | 0.1617 | 0.2086 | 80.58s | 1.181 | -23712.5ms |
| current-mix | slo_no_preemption_lookup | 3 | 1 | 6.25% | 0.1975 | 0.2107 | 59.57s | 1.129 | 10260.2ms |
| current-mix | slo_no_preemption_lookup | 3.5 | 1 | 0.00% | 0.2175 | 0.2175 | 59.45s | 1.099 | 22046.4ms |
| current-mix | slo_no_preemption_lookup | 4 | 1 | 2.50% | 0.2083 | 0.2136 | 59.76s | 1.135 | 27390.5ms |
| large-heavy | current | 2.5 | 1 | 72.50% | 0.0507 | 0.1843 | 128.14s | 1.387 | -58828.3ms |
| large-heavy | current | 3 | 1 | 86.25% | 0.0174 | 0.1188 | 258.23s | 1.366 | -193177.7ms |
| large-heavy | current | 3.5 | 1 | 51.25% | 0.0816 | 0.1675 | 154.65s | 1.333 | -70208.4ms |
| large-heavy | current | 4 | 1 | 48.75% | 0.0788 | 0.1537 | 193.28s | 1.389 | -76447.4ms |
| large-heavy | slo_no_preemption_lookup | 2.5 | 1 | 32.50% | 0.1149 | 0.1702 | 153.66s | 1.447 | -72781.5ms |
| large-heavy | slo_no_preemption_lookup | 3 | 1 | 32.50% | 0.1239 | 0.1836 | 136.11s | 1.412 | -74044.2ms |
| large-heavy | slo_no_preemption_lookup | 3.5 | 1 | 21.25% | 0.1364 | 0.1733 | 220.61s | 1.423 | -139716.8ms |
| large-heavy | slo_no_preemption_lookup | 4 | 1 | 21.25% | 0.1531 | 0.1945 | 130.77s | 1.490 | -55776.9ms |
| rectangular-mix | current | 2.5 | 1 | 53.75% | 0.0857 | 0.1853 | 115.05s | 1.026 | -61643.9ms |
| rectangular-mix | current | 3 | 1 | 46.25% | 0.0892 | 0.1659 | 128.93s | 1.027 | -67337.5ms |
| rectangular-mix | current | 3.5 | 1 | 47.50% | 0.0772 | 0.1452 | 204.21s | 1.013 | -140793.4ms |
| rectangular-mix | current | 4 | 1 | 22.50% | 0.1460 | 0.1884 | 103.44s | 1.045 | -30721.9ms |
| rectangular-mix | slo_no_preemption_lookup | 2.5 | 1 | 17.50% | 0.1635 | 0.1982 | 68.94s | 1.036 | -18848.4ms |
| rectangular-mix | slo_no_preemption_lookup | 3 | 1 | 18.75% | 0.1659 | 0.2042 | 110.42s | 1.054 | -48866.8ms |
| rectangular-mix | slo_no_preemption_lookup | 3.5 | 1 | 13.75% | 0.1605 | 0.1860 | 94.53s | 1.064 | -27467.7ms |
| rectangular-mix | slo_no_preemption_lookup | 4 | 1 | 6.25% | 0.1797 | 0.1917 | 71.93s | 1.057 | 3234.6ms |
