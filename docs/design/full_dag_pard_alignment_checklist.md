# Full DAG Runtime + Full PARD Alignment Checklist

更新时间：2026-06-03

本文档是 Gate A 的验收清单，用来把飞书文档、飞书评论、白板和 PARD 论文要求映射到后续实现、测试和实验。所有条目都是必选项；任何条目无法确认时，必须暂停并向正刚确认，不能用简化实现绕过。

参考：

- 飞书文档：`GCZawDo4EikVvmkH06vcEWrhnBZ`
- 飞书评论文档：`QTStd1RfjoItenxArOYcauPbnve`
- 本地白板图片：`feishu_assets/dag_role_system.jpg`
- PARD 论文：`PARD: Enhancing Goodput for Inference Pipeline via Proactive Request Dropping`
- 执行计划：`full_dag_runtime_pard_execution_plan.md`

## 1. 飞书要求对齐

| 要求 | 必须实现的内容 | 验收方式 |
|---|---|---|
| DAG 小角色系统 | text encoder、image encoder、VAE encoder、DiT denoise、VAE decoder、audio decoder 都注册为 `DagStageSpec` | `tests/engine/test_dag_runtime.py` 覆盖 stage 拓扑、入口、出口、branch/merge |
| 资源配比 | 每类 stage 配置 replica 数和每个 replica 卡数 | `DagStageResourceSpec` 单测；若飞书白板未给出最终配比，向正刚确认 |
| StagePool 简化 | StagePool 不再堆叠零散规则，而是通过目标函数选择 replica | 目标函数输出 score 和分项解释；subagent review |
| 目标函数 | score 显式包含 goodput、miss risk、token work、replica load balance、optimal batch size / cost | `tests/engine/test_dag_stage_scheduler.py` 覆盖 score 分量和 tie-break |
| token 粒度 | DiT stage 继续使用第一阶段 token objective / step-preemptive scheduler | policy alias 不覆盖第一阶段 alias；token work profile 可见 |
| step 边界抢占 | 只允许在 denoise step 边界抢占、drop、恢复；不在单个 step 内中断 kernel | `tests/engine/test_pard_cleanup.py` 覆盖 running-step-boundary |
| 请求 push | request 控制面到达后立即由 orchestrator push 到 DAG runtime | request trace 记录 stage arrival 和 push path |
| embedding cache pull | cache 数据面以 handle/location/size 表达，consumer stage 可 pull | `tests/engine/test_dag_cache.py` 覆盖 pull |
| prefetch | runtime 根据下游 stage 预测提前 prefetch cache | trace 记录 prefetch start/end/hit/miss |
| stage 状态上报 | 每个 stage 上报队列长度、当前 batch、剩余 token work、预计执行时间、cache readiness、stage 间传输状态 | `snapshot()` 接口和 metric CSV 覆盖 |
| 跨请求流水重叠 | 支持白板里的 request `i+1` encode、request `i` DiT、request `i-1` decode 同时存在于不同 stage | fake DAG simulator 和真实 smoke trace 中能看到跨 stage overlap |
| 资源比例搜索 | 资源配比不能只靠手填固定值，需要支持 offline sampling + objective function 评估候选配比 | 资源配置实验记录候选 replica/card、目标函数 score 和选择原因 |
| 实验安全包 | 每轮实验只保留 README、CSV、SVG、manifest | 结果包 subagent review；不包含 raw/log/jsonl/trace/profile/大文件 |

## 2. PARD 论文机制对齐

| PARD 机制 | 必须实现的内容 | 验收方式 |
|---|---|---|
| pipeline module | 每个 DAG stage 对应一个 PARD module | `DagStageSpec.stage_id` 与 PARD module id 一一对应 |
| worker | 每个 stage replica 对应 PARD worker | `DagStageReplicaSpec` 和 StagePool snapshot 覆盖 |
| State Planner | 对每个 stage 入口估计 `L_pre + L_cur + L_sub` | `tests/engine/test_pard_planner.py` |
| `L_pre` | 请求从 root arrival 到当前 stage arrival 的已消耗时间 | request context trace |
| `L_cur` | 当前 stage 的 queue wait、batch wait、execution duration | planner 单测和 stage metric |
| `L_sub` | 后续 DAG 路径的 queue wait、execution duration、batch wait 分位估计 | branch/merge DAG 单测；多路径取最大估计 |
| `lambda` | 支持 PARD 默认值和 lower/upper ablation | `dag_pard_lower`、`dag_pard_upper` policy |
| Request Broker | stage 入口根据端到端估计决定进入队列或 proactive drop | `tests/engine/test_pard_broker.py` |
| proactive drop | 预测必然超出 SLO 的请求在 stage 入口主动丢弃 | drop trace 记录 stage/reason/estimate |
| reactive drop | 已错过 SLO 或已不可能恢复的请求按 reactive policy 丢弃 | `dag_reactive_drop` ablation |
| DEPQ | 按剩余 latency budget 维护双端优先队列 | `tests/engine/test_pard_priority.py` 覆盖 min/max pop |
| HBF | 高负载时优先保留高 budget 请求 | `mu > 1 + eps` 单测 |
| LBF | 正常负载时优先处理低 budget 请求 | `mu < 1 - eps` 单测 |
| delayed transition | `mu` 在 `[1 - eps, 1 + eps]` 内保持当前 priority mode | hysteresis 单测 |
| workload intensity | 计算 `mu = T_in / T_m`；`T_in` 来自平滑输入工作量窗口，`T_m` 来自 per-stage throughput profile | trace 输出 `mu`、`T_in`、`T_m`、平滑窗口和 profile version |
| `eps` | `eps` 来自近期输入工作量与平滑输入工作量的偏差，用于 delayed transition | priority trace 输出 `eps` 和 transition reason |
| invalid compute | 已消耗但最终 miss/drop 的计算量进入 invalid compute | metric CSV 和 README 汇总 |
| ablation | no-drop、reactive、fixed、lower、upper、FCFS、HBF、LBF、instant、full PARD 都是独立 policy | runner policy allowlist 和实验矩阵覆盖 |
| paper-native ablation | 每个 PARD 论文原生 ablation 都必须有独立 local alias、独立语义和独立实验项，不能合并命名 | README 和 CSV 同时输出本地 policy alias、paper-native name、ablation semantic |

Paper-native ablation 映射表：

| Paper name | Local alias | 必须表达的语义 |
|---|---|---|
| `PARD-back` | `dag_pard_back` | 回退式后验丢弃 / backward baseline，对比 proactive drop 的收益 |
| `PARD-sf` | `dag_pard_sf` | static fixed priority / 固定优先级 baseline |
| `PARD-oc` | `dag_pard_oc` | oracle control / 最优或近似 oracle 控制 baseline，用于界定上限 |
| `PARD-split` | `dag_pard_split` | 固定资源切分 baseline |
| `PARD-WCL` | `dag_pard_wcl` | workload-class-aware 或论文定义的 WCL baseline，语义必须按论文逐项对齐 |
| `PARD-lower` | `dag_pard_lower` | planner lower-bound ablation |
| `PARD-upper` | `dag_pard_upper` | planner upper-bound ablation |
| `PARD-FCFS` | `dag_pard_fcfs` | Request Broker 使用 FCFS priority |
| `PARD-HBF` | `dag_pard_hbf` | 固定 High Budget First |
| `PARD-LBF` | `dag_pard_lbf` | 固定 Low Budget First |
| `PARD-instant` | `dag_pard_instant` | instant transition，不使用 delayed transition |
| `full PARD` | `dag_full_pard` | State Planner + Request Broker + DEPQ + HBF/LBF adaptive priority + delayed transition + cleanup |

## 3. 代码模块映射

| 模块 | 目标文件 | 主要责任 |
|---|---|---|
| DAG 类型 | `vllm_omni/engine/dag_types.py` | stage spec、request context、resource spec、stage input/output |
| DAG runtime | `vllm_omni/engine/dag_runtime.py` | DAG 拓扑校验、request 生命周期、stage 转移 |
| stage adapter | `vllm_omni/engine/stage_adapters.py` | text/image/VAE encoder、DiT、VAE/audio decoder 真实入口 |
| cache | `vllm_omni/engine/dag_cache.py` | cache handle、pull、prefetch、trace |
| PARD planner | `vllm_omni/engine/pard_planner.py` | `L_pre + L_cur + L_sub` 估计 |
| PARD broker | `vllm_omni/engine/pard_broker.py` | proactive/reactive drop、broker queue |
| PARD priority | `vllm_omni/engine/pard_priority.py` | DEPQ、HBF/LBF、delayed transition |
| PARD runtime | `vllm_omni/engine/pard_runtime.py` | planner/broker/runtime glue、metrics |
| StagePool | `vllm_omni/engine/stage_pool.py` | stage-level objective function 和 replica selection |
| DiT scheduler | `vllm_omni/diffusion/sched/token_slo_step_scheduler.py` | token objective、step boundary preemption/drop |
| orchestrator | `vllm_omni/engine/orchestrator.py` | request push、DAG runtime 接线、client-visible 状态 |
| async engine | `vllm_omni/engine/async_omni_engine.py` | 对外请求入口、DAG policy 接入 |
| benchmark | `benchmarks/diffusion/e2e_slo_frontier_experiment.py`、`benchmarks/diffusion/run_910b_e2e_slo_frontier.sh` | policy allowlist、metrics、GitHub-safe package |

## 4. 测试矩阵

| 测试文件 | 必须覆盖 |
|---|---|
| `tests/engine/test_dag_runtime.py` | DAG 拓扑、无环校验、branch/merge、request lifecycle |
| `tests/engine/test_dag_stage_adapters.py` | 六类 stage adapter 的 submit/cancel/cleanup/snapshot wiring |
| `tests/engine/test_dag_resource_model.py` | replica/card 配置、资源配比校验 |
| `tests/engine/test_dag_cache.py` | cache handle、pull、prefetch、trace |
| `tests/engine/test_pard_planner.py` | `L_pre`、`L_cur`、`L_sub`、lambda、estimate error |
| `tests/engine/test_pard_broker.py` | proactive drop、reactive drop、stage entry gate |
| `tests/engine/test_pard_priority.py` | DEPQ、HBF、LBF、delayed transition、burst/steady load |
| `tests/engine/test_pard_cleanup.py` | queued/batched/running-step-boundary/completed cleanup |
| `tests/engine/test_pard_metrics.py` | drop stage、drop reason、invalid compute、estimate error |
| `tests/engine/test_dag_stage_scheduler.py` | StagePool objective function、score 分项、tie-break |

## 5. Gate 依赖

| Gate | 不可跳过的依赖 |
|---|---|
| Gate A | 第一阶段完整实验结果包与旧结果对比完成；本 checklist 通过 subagent review |
| Gate B | Gate A 通过后，只实现 full DAG runtime 基础，不写 PARD drop 逻辑 |
| Gate C | Gate B 通过后，才实现 State Planner、Request Broker、DEPQ、HBF/LBF；所有 stage profile/lookup 必须采集并验证误差 |
| Gate D | Gate C 通过后，才实现 drop/cancel/cleanup、step 边界处理、真实 910B full DAG smoke |
| Gate E | Gate D 通过后，才启动性能实验和 PARD ablation |

## 6. 禁止项

以下做法不允许作为 Gate 通过依据：

1. 不允许只接入 DiT stage 后声称完成 full DAG runtime。
2. 不允许用 encoder、decoder 或 audio decoder 的固定常数延迟代替真实 profile/lookup。
3. 不允许跳过 text encoder、image encoder、VAE encoder、VAE decoder 或 audio decoder。
4. 不允许用 fake DAG simulator 或无 NPU 单测替代真实 910B full DAG smoke。
5. 不允许只实现 proactive drop 而缺少 DEPQ、HBF/LBF、自适应 priority 和 delayed transition。
6. 不允许没有 drop/cancel/cleanup trace 就运行性能矩阵。

## 7. 硬阻塞项

遇到以下任一情况必须暂停并向正刚确认：

1. 飞书白板中的最终 stage 资源配比不明确。
2. full-DAG smoke 的目标模型和拓扑不明确。
3. audio decoder 的模型路径、权重或执行入口不明确。
4. text/image/VAE encoder、VAE decoder 的真实入口无法定位。
5. 任一 stage 缺少可用 profile/lookup 数据。
6. 任一 stage 的 cancel/cleanup 语义无法从代码确认。
7. PARD 官方代码或 artifact 可用但尚未确认是否作为实现对齐对象。
8. 910B 环境缺少 full DAG smoke 所需 runtime、权重或权限。
9. embedding cache pull/prefetch 的 ownership、trace schema 或 cache handle 生命周期不明确。

## 8. 实验验收

full DAG/PARD 性能实验只能在 Gate D 通过后启动。每轮结果必须对比：

Baseline policies：

- `current`
- `pr4024_dynamic`
- `slo_no_preemption_token_objective`
- `dag_no_drop`
- `dag_reactive_drop`

Paper-native PARD ablations：

- `dag_pard_back`
- `dag_pard_sf`
- `dag_pard_oc`
- `dag_pard_split`
- `dag_pard_wcl`
- `dag_pard_lower`
- `dag_pard_upper`
- `dag_pard_fcfs`
- `dag_pard_hbf`
- `dag_pard_lbf`
- `dag_pard_instant`

Full policies：

- `dag_full_pard`
- `dag_full_pard_step_preemptive`

必须汇总：

- miss rate
- goodput
- throughput
- P95
- proactive drop rate
- reactive drop rate
- invalid compute
- drop stage distribution
- estimate error
- queue wait
- batch wait
- cache pull/prefetch latency
- replica load distribution
- step preemption count
