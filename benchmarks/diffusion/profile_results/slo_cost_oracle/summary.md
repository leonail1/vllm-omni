# SLO Cost Oracle Replay Results

环境：910B，本次为 scheduler replay，不启动真实模型；denoise step cost 来自 offline profile lookup/formula。

公共配置：Qwen/Qwen-Image，50 denoise steps，max_num_seqs=4，mixed_slo workload，StepScheduler(FIFO) vs SloStepScheduler。

| 场景 | FIFO misses | SLO misses | 违约下降 | FIFO throughput | SLO throughput | 吞吐变化 | FIFO mean batch | SLO mean batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ia=12.0, tight=3.5, loose=20.0 | 52 | 12 | 76.9% | 0.06940 | 0.07002 | +0.89% | 1.97 | 2.32 |
| ia=12.0, tight=4.0, loose=20.0 | 49 | 9 | 81.6% | 0.06940 | 0.07015 | +1.07% | 1.97 | 2.35 |
| ia=14.0, tight=3.5, loose=20.0 | 26 | 3 | 88.5% | 0.06887 | 0.06813 | -1.08% | 1.91 | 1.99 |

## 结论

- ia=12, tight=3.5, loose=20：违约从 52 降到 12，下降 76.9%，吞吐提升 0.89%。
- ia=12, tight=4.0, loose=20：违约从 49 降到 9，下降 81.6%，吞吐提升 1.07%。
- ia=14, tight=3.5, loose=20：违约从 26 降到 3，下降 88.5%，吞吐下降 1.08%，说明较低压力下策略更偏向保 SLO。

## 复现命令

```bash
.venv/bin/python benchmarks/diffusion/slo_scheduler_replay.py --num-requests 80 --interarrival-s 12.0 --max-num-seqs 4 --steps 50 --workload mixed_slo --tight-slo-scale 3.5 --loose-slo-scale 20.0 --batch-growth-alpha 0.6 --model Qwen/Qwen-Image --step-cost-model-path benchmarks/diffusion/profile_results/latent_token_aspect/step_cost_model.json --step-cost-metric p90_ms --step-cost-formula qwen_image_910b_tp2_v1 --output-file benchmark_outputs/slo_cost_oracle_final/mixed_slo_ia12_t35_l20.json
.venv/bin/python benchmarks/diffusion/slo_scheduler_replay.py --num-requests 80 --interarrival-s 12.0 --max-num-seqs 4 --steps 50 --workload mixed_slo --tight-slo-scale 4.0 --loose-slo-scale 20.0 --batch-growth-alpha 0.6 --model Qwen/Qwen-Image --step-cost-model-path benchmarks/diffusion/profile_results/latent_token_aspect/step_cost_model.json --step-cost-metric p90_ms --step-cost-formula qwen_image_910b_tp2_v1 --output-file benchmark_outputs/slo_cost_oracle_final/mixed_slo_ia12_t40_l20.json
.venv/bin/python benchmarks/diffusion/slo_scheduler_replay.py --num-requests 80 --interarrival-s 14.0 --max-num-seqs 4 --steps 50 --workload mixed_slo --tight-slo-scale 3.5 --loose-slo-scale 20.0 --batch-growth-alpha 0.6 --model Qwen/Qwen-Image --step-cost-model-path benchmarks/diffusion/profile_results/latent_token_aspect/step_cost_model.json --step-cost-metric p90_ms --step-cost-formula qwen_image_910b_tp2_v1 --output-file benchmark_outputs/slo_cost_oracle_final/mixed_slo_ia14_t35_l20.json
```
