# Qwen-Image v0.23.0rc1 stacked PR 对比实验

模型：`/dataset/models/stage_transfer_profile_models/Qwen_Qwen-Image`

本次实验验证四组路径：

| 来源 | 模式 | step_execution | stage_split |
| --- | --- | --- | --- |
| clean | forward | false | false |
| PR1 | forward | false | false |
| PR2 | forward | false | false |
| PR2 | stage_step | true | true |

clean baseline 使用 `/home/lzg/vllm-omni-v023rc1-original-baseline-20260626`，revision `7b837944`。PR1 使用 `/home/lzg/vllm-omni-v023rc1-pr1-pipeline-20260629`，revision `d3dd1352`。PR2 使用 `/home/lzg/vllm-omni-v023rc1-pr2-stage-20260629`，revision `d13fb927`。

每个 case 执行 `1` 次 warmup 和 `3` 次 measured repeat。warmup 只用于触发首次编译/缓存，不计入 measured 统计。像素 diff 使用第 1 次 measured 输出计算；实验目录只保留 JSON 和 README，不保留图片。

## 复现入口

```bash
cd /home/lzg/vllm-omni-v023rc1-pr2-stage-20260629
bash experiments/qwen_smoke/run_v023_stage_step_compare.sh
```

## 保留文件

| 文件 | 用途 |
| --- | --- |
| `README.md` | 中文实验说明和结论 |
| `v023_pixel_diff.json` | 像素差异结果 |
| `v023_run_metrics.json` | 每个 case 的耗时、代码来源和运行参数 |
| `v023_memory_metrics.json` | 每个 case 的 NPU HBM/AICore 采样汇总 |
| `run_v023_stage_step_compare.sh` | clean/PR1/PR2 对比入口 |
| `npu_memory_sampler.py` | NPU HBM/AICore 采样器 |
| `compare_pixel_diff.py` | 像素差异计算脚本 |
| `run_omni_smoke.py` | 单个 case 的 smoke runner |
| `qwen_image_stage_split_3stage.yaml` | stage_step 配置 |

## 运行耗时

| 步数 | 来源 | 模式 | step_execution | stage_split | 成功 | 初始化 s | warmup 生成 s | measured 平均 s | 相对 clean forward | 总耗时 s |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | clean | forward | false | false | true | 63.0619 | 2.1778 | 2.1413 | +0.00% | 73.7011 |
| 4 | pr1 | forward | false | false | true | 62.8388 | 2.1783 | 2.1337 | -0.36% | 73.4602 |
| 4 | pr2 | forward | false | false | true | 62.8140 | 2.2683 | 2.1378 | -0.17% | 73.5400 |
| 4 | pr2 | stage_step | true | true | true | 110.6910 | 66.8597 | 2.2298 | +4.13% | 186.2977 |
| 20 | clean | forward | false | false | true | 63.6030 | 9.4254 | 9.4147 | +0.00% | 103.2475 |
| 20 | pr1 | forward | false | false | true | 62.2319 | 9.4075 | 9.4000 | -0.16% | 101.8244 |
| 20 | pr2 | forward | false | false | true | 62.7536 | 9.4526 | 9.4476 | +0.35% | 102.5310 |
| 20 | pr2 | stage_step | true | true | true | 110.9287 | 73.9846 | 9.5468 | +1.40% | 215.5416 |

## NPU 显存和利用率

`主动设备` 格式为 `device:峰值增量MB/峰值占用MB`，峰值增量以采样开始时的 HBM 为基线。

| 步数 | 来源 | 模式 | 主动设备 | 最大 AICore % |
| ---: | --- | --- | --- | ---: |
| 4 | clean | forward | 0:61112MB/64492MB | 62 |
| 4 | pr1 | forward | 0:61115MB/64496MB | 62 |
| 4 | pr2 | forward | 0:61114MB/64495MB | 62 |
| 4 | pr2 | stage_step | 1:40981MB/44357MB, 2:24379MB/27751MB, 0:16208MB/19589MB | 62 |
| 20 | clean | forward | 0:61113MB/64492MB | 63 |
| 20 | pr1 | forward | 0:61115MB/64496MB | 63 |
| 20 | pr2 | forward | 0:61115MB/64496MB | 63 |
| 20 | pr2 | stage_step | 1:40980MB/44357MB, 2:24379MB/27752MB, 0:16209MB/19590MB | 63 |

## PR2 stage_step 按 role 显存

| 步数 | role | device | 初始 HBM MB | 峰值 HBM MB | 峰值增量 MB | 最大 AICore % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | encode | 0 | 3381 | 19589 | 16208 | 0 |
| 4 | dit | 1 | 3376 | 44357 | 40981 | 62 |
| 4 | decode | 2 | 3372 | 27751 | 24379 | 0 |
| 20 | encode | 0 | 3381 | 19590 | 16209 | 0 |
| 20 | dit | 1 | 3377 | 44357 | 40980 | 63 |
| 20 | decode | 2 | 3373 | 27752 | 24379 | 1 |

## 像素差异

| 对比 | max_abs | rmse | nonzero_pixels | nonzero_values |
| --- | ---: | ---: | ---: | ---: |
| clean_vs_pr1_forward_4step | 0 | 0.000000 | 0 | 0 |
| clean_vs_pr2_forward_4step | 0 | 0.000000 | 0 | 0 |
| clean_vs_pr2_stage_step_4step | 0 | 0.000000 | 0 | 0 |
| pr1_forward_vs_pr2_forward_4step | 0 | 0.000000 | 0 | 0 |
| pr2_forward_vs_stage_step_4step | 0 | 0.000000 | 0 | 0 |
| clean_vs_pr1_forward_20step | 0 | 0.000000 | 0 | 0 |
| clean_vs_pr2_forward_20step | 0 | 0.000000 | 0 | 0 |
| clean_vs_pr2_stage_step_20step | 0 | 0.000000 | 0 | 0 |
| pr1_forward_vs_pr2_forward_20step | 0 | 0.000000 | 0 | 0 |
| pr2_forward_vs_stage_step_20step | 0 | 0.000000 | 0 | 0 |

结论：clean、PR1 forward、PR2 forward 和 PR2 stage_step 均成功生成图片。若像素差异全为 0，则说明 pipeline/runner 重构和 stage-step 拆分均未改变输出；性能判断以 measured 平均为准，cold path 的初始化和 warmup 单独列出。
