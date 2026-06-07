# 阶段性汇报：ARCH_SPEC 架构级 TSP-GLS 自动算法设计框架与 ReEVO 对比

> 汇报对象：当前 `eohagent` 框架阶段性结果  
> 主要实验目录：`eohagent/results/tsp_gls_arch_spec_deepseek_10x20_v1`  
> ReEVO 对比目录：`eohagent/results/baselines/reevo_deepseek_10x200_v1`  
> 生成时间：2026-06-02

---

## 0. 一句话总结

当前框架已经从“让 LLM 直接改一个启发式函数”升级为“让 LLM 像专家一样规划、执行、审计一组受约束的 GLS 架构级实验”。

在 TSP-GLS 任务上，当前 `ARCH_SPEC` 框架在 20 代、每代 10 个候选的设置下，训练集最优 gap 从初始种子的 `0.18540` 降到 `0.00437`；在 TSPLib 测试集上，最终种群最优平均 gap 为 `0.21278`。同一训练/评估接口下，ReEVO 当前对比实验的 TSPLib 最优平均 gap 为 `0.34209`。

因此，阶段性结论是：

```text
我们的 ARCH_SPEC 架构级搜索框架目前明显优于这一次适配后的 ReEVO baseline。
```

但需要强调：这次 ReEVO 对比不是 ReEVO 论文原始设置，而是为了和我们当前实验公平比较而做的“同 evaluator / 同训练集 / 同 API / 同样本预算附近”的适配版本。因此它可以作为阶段性 baseline，但不能直接解读为“完全复现并超过 ReEVO 论文结果”。

---

## 1. 当前框架到底在做什么？

### 1.1 研究目标

我们的目标不是简单让 LLM 写一个函数，而是让 LLM 模仿人类专家设计算法的过程：

1. 观察已有算法表现；
2. 提出机制层面的假设；
3. 设计一组受控实验；
4. 每个实验只改变一个明确组件；
5. 运行正式评估；
6. 总结哪些机制有效、哪些失败；
7. 下一代基于这些经验继续设计。

这个过程对应到当前代码，就是：

```text
expert seeds
  -> generation planner
  -> controlled ARCH_SPEC child specifications
  -> deterministic code generation
  -> contract check
  -> formal evaluator
  -> training diagnostics
  -> quality-diversity survival
  -> generation reflection / expert notebook
  -> next generation
```

---

## 2. 为什么要从单函数搜索升级到 ARCH_SPEC 架构级搜索？

早期的 TSP-GLS 自动设计主要让 LLM 修改一个函数，例如 `update_edge_distance`。这种方式有几个问题：

1. 搜索空间不透明：LLM 改了什么机制很难审计；
2. 代码变化粒度不稳定：有时一次修改会混合多个因素；
3. 难以做受控实验：无法保证“只改变一个机制”；
4. 难以表达真正的人类专家式设计：专家通常不是只改一行公式，而是组合设计距离更新、惩罚策略、扰动策略、重启策略、调度策略。

因此当前框架改为 `ARCH_SPEC` 结构化搜索。每个候选不是任意代码，而是一个可审计的 GLS 架构规范。

---

## 3. ARCH_SPEC 搜索空间

当前每个候选包含 5 个固定开放层：

```text
distance_update_policy
penalty_policy
escape_policy
restart_policy
schedule_policy
```

### 3.1 distance_update_policy

负责决定 GLS 中如何根据边的使用情况、边长、utility 等信号更新边距离。

可选类型包括：

```text
linear_usage
sqrt_usage
log_usage
length_weighted_usage
utility_ratio
rank_usage
smooth_saturation
```

该层控制的典型参数包括：

```text
type
strength
usage_power
length_weight
cap_multiplier
```

### 3.2 penalty_policy

负责决定哪些边被加惩罚，以及惩罚如何增长/衰减。

典型参数：

```text
edge_selection: top_gap / tour_utility / rank_topk / mixed_gap_utility
top_k
increment
decay
```

### 3.3 escape_policy

负责扰动/逃逸策略，例如固定扰动、周期触发、停滞触发、规模自适应触发。

典型参数：

```text
trigger
moves
period
stagnation_threshold
size_threshold
max_moves
```

### 3.4 restart_policy

负责当搜索停滞或周期触发时如何重启或处理 penalty。

典型参数：

```text
action: none / reset_to_best / reset_to_initial / decay_penalty / rebuild_nearest
trigger: never / periodic / stagnation
period
stagnation_threshold
penalty_decay
```

### 3.5 schedule_policy

负责动态调度距离更新强度，例如早期/后期不同强度，停滞时增强等。

典型参数：

```text
strength_schedule: constant / decay_after_early / stagnation_boost / size_scaled
early_iterations
late_multiplier
stagnation_multiplier
size_threshold
size_multiplier
```

---

## 4. 当前框架的完整工作流

### 4.1 初始化：专家种子 bootstrap

本实验使用 10 个人工设计的 ARCH_SPEC seed，而不是随机从空白开始。

配置：

```yaml
bootstrap:
  use_baseline_seed: true
  n_proc: 10
  seed_path: ./eohagent/seeds/tsp_gls_arch_spec_baseline.json
```

初始化评估结果如下：

| seed | training objective |
|---|---:|
| spec_linear_top12_standard_seed | 0.18540 |
| spec_linear_top8_periodic_escape_seed | 0.38975 |
| spec_linear_top8_standard_seed | 0.44111 |
| spec_linear_top8_stagnation_restart_seed | 0.44111 |
| spec_length_top8_standard_seed | 0.51305 |
| spec_linear_top8_decay_seed | 0.77562 |
| spec_linear_top3_standard_seed | 1.70628 |
| spec_utility_top8_standard_seed | 2.31325 |
| spec_rank_top8_standard_seed | 2.77272 |
| spec_smooth_top8_standard_seed | 2.80635 |

初始化后种群最好是：

```text
best seed = spec_linear_top12_standard_seed
training objective = 0.18540
```

注意：这个专家种子初始化本身是我们框架的重要组成部分。它让搜索从一组可解释、可比较、机制不同的启发式结构开始，而不是完全随机生成代码。

---

### 4.2 每代先生成 generation plan

每一代开始时，LLM 不直接写代码，而是先作为“实验设计者”输出一份 JSON 计划。

这个计划必须包含：

```text
generation_goal
avoid_repeating
offspring_plans
```

每个 offspring plan 必须包含：

```text
offspring_index
parent_id
group_id
variant_id
group_hypothesis
controlled_component
delta
preserve
avoid
expected_effect
child_spec
```

其中最重要的是：

```text
child_spec 必须是完整 ARCH_SPEC
child_spec 必须相对 parent 只改变 controlled_component 这一层
```

这使得每个候选都对应一个明确的“机制实验”，而不是不可解释的代码修改。

---

### 4.3 单层变更审计

当前框架对每个 child_spec 做严格检查：

```text
child_spec changed layers == [controlled_component]
```

如果一个候选同时改了两个层，例如同时改 `penalty_policy` 和 `escape_policy`，则直接拒绝。

这个机制非常重要，因为它保证了实验可归因性：

```text
如果候选变好或变坏，我们知道是哪一层的哪一个 delta 导致的。
```

这也是当前框架模仿人类专家实验设计的关键部分。

---

### 4.4 由 ARCH_SPEC 确定性生成代码

当前 ARCH_SPEC 不是直接让 LLM 写完整 Python 类，而是由系统确定性生成：

```python
class CandidateGLSArchitecture(SpecCandidateGLSArchitecture):
    def __init__(self):
        super().__init__(ARCH_SPEC)
```

也就是说，LLM 负责设计结构化 spec，系统负责把 spec 转成代码。

这样做有几个好处：

1. 减少语法错误；
2. 避免 LLM 写出不可审计的逻辑；
3. 保证候选都在同一个架构框架内比较；
4. 让搜索更接近“算法结构设计”，而不是“代码碰运气”。

---

### 4.5 contract check 和 formal evaluator

候选生成后，会先进行代码 contract check，再进入正式 evaluator。

正式 evaluator 设置：

```yaml
evaluator:
  n_inst_eva: 64
  time_limit: 60
  ite_max: 1000
  perturbation_moves: 1
  timeout: 500
```

目标是最小化 64 个训练实例的平均 gap：

```text
objective = average training gap
lower is better
```

---

### 4.6 training diagnostics

每个候选不仅记录平均 objective，还记录更细的训练诊断，例如：

```text
mean_gap
std_gap
tail_gap
max_gap
failure_signature
worst_training_instances
mean_delta_vs_parent
tail_delta_vs_parent
```

这些信息会反馈给下一代 planner 和 reflection。

这一步非常关键，因为人类专家不仅看平均值，也会问：

```text
这个算法是不是只在少数实例上崩了？
尾部风险变大了吗？
失败模式是什么？
相对父代到底改善了哪些实例？
```

当前框架已经开始具备这种诊断能力。

---

### 4.7 quality-diversity survival selection

每一代会把旧种群和新 offspring 合并，然后通过 `quality_diversity` 选择保留 10 个个体。

当前 survival selection 不只是简单按平均 objective 排序，而是考虑：

```text
objective
tail_gap
max_gap
std_gap
ARCH_SPEC niche
```

这使得最终种群不一定只保留训练分数最好的克隆，而会保留一些训练略差但结构不同的候选。

这点在本次实验中非常重要，因为最终 TSPLib 最好的个体并不是训练 objective 最好的个体。

---

### 4.8 generation reflection 与 expert notebook

每代结束后，LLM 会总结本代实验：

```text
validated_principles
failed_hypotheses
supported_deltas
refuted_deltas
failure_patterns
next_generation_priorities
open_questions
```

这些内容会写入：

```text
expert_notebook.json
mechanism_archive.json
compact_context.json
session_log.jsonl
```

这样下一代不是从零开始，而是带着“实验记忆”继续搜索。

这对应人类专家的过程：

```text
做完一轮实验 -> 总结经验 -> 下一轮不要重复失败方向 -> 深挖有效机制
```

---

### 4.9 fail-closed / no fallback

当前框架明确禁止 fallback。

如果出现：

```text
planner JSON 错误
parent 不存在
child_spec 不合法
候选重复
contract check 失败
formal evaluator 失败
reflection 失败
```

系统不会自动替换成默认计划，也不会编造候选，而是直接拒绝该候选或停止该代。

本次 20 代实验中：

```text
generation_failed = 0
planner 每代成功
reflection 每代成功
repair_used = 0
```

候选失败统计：

```text
总计划 offspring = 200
成功正式评估 = 178
失败/拒绝 = 22
```

失败原因：

```text
Duplicate ARCH_SPEC code generated: 16
Code evaluation failed: 6
```

这说明当前实验流程比较干净：失败候选没有被隐藏替换，也没有污染结果。

---

## 5. 主实验配置

### 5.1 LLM 配置

```yaml
model: deepseek-v4-flash
api_base: https://api.deepseek.com
temperature: 0.7
max_retries: 3
timeout: 600
thinking: disabled
```

### 5.2 Evolution 配置

```yaml
pop_size: 10
generations: 20
offspring_per_generation: 10
n_proc: 5
parent_selection: rank
survival_selection: quality_diversity
allow_archive_parents: false
deduplicate_code: true
deduplicate_objective: false
```

### 5.3 Evaluator 配置

```yaml
n_inst_eva: 64
time_limit: 60
ite_max: 1000
perturbation_moves: 1
timeout: 500
objective_direction: minimize
```

---

## 6. 训练过程结果

### 6.1 每代 best / median / worst objective

| generation | best | median | worst |
|---:|---:|---:|---:|
| 0 | 0.18540 | 0.64434 | 2.80635 |
| 1 | 0.15471 | 0.21970 | 0.51305 |
| 2 | 0.15471 | 0.16395 | 0.19168 |
| 3 | 0.15471 | 0.15471 | 0.15625 |
| 4 | 0.15471 | 0.15471 | 0.15600 |
| 5 | 0.13733 | 0.15471 | 0.15471 |
| 6 | 0.13435 | 0.15471 | 0.15655 |
| 7 | 0.13435 | 0.13733 | 0.15471 |
| 8 | 0.12394 | 0.13584 | 0.14446 |
| 9 | 0.12394 | 0.12394 | 0.13435 |
| 10 | 0.07947 | 0.12394 | 0.13435 |
| 11 | 0.00479 | 0.12294 | 0.12394 |
| 12 | 0.00479 | 0.06069 | 0.12194 |
| 13 | 0.00479 | 0.04080 | 0.08498 |
| 14 | 0.00456 | 0.00915 | 0.03968 |
| 15 | 0.00456 | 0.00530 | 0.00745 |
| 16 | 0.00456 | 0.00479 | 0.00916 |
| 17 | 0.00456 | 0.00479 | 0.00916 |
| 18 | 0.00456 | 0.00479 | 0.00916 |
| 19 | 0.00437 | 0.00479 | 0.00916 |
| 20 | 0.00437 | 0.00456 | 0.00916 |

### 6.2 训练曲线解读

训练过程可以分成几个阶段。

#### 阶段 1：快速淘汰差种子（generation 0 -> 3）

```text
gen0 best = 0.18540
gen1 best = 0.15471
gen3 best = 0.15471
```

第一代已经找到比所有 bootstrap seeds 更好的候选 `h_0009`。

`h_0009` 的关键变化是：

```text
escape_policy.moves = 3
escape_policy.max_moves = 3
trigger = periodic
period = 40
```

说明早期搜索中，逃逸/扰动强度对 GLS 有明显影响。

#### 阶段 2：稳定围绕 0.12-0.15 改进（generation 4 -> 10）

```text
gen5 best = 0.13733
gen8 best = 0.12394
gen10 best = 0.07947
```

此阶段逐渐加入：

```text
penalty decay = 0.1
stagnation escape
utility_ratio distance update
```

其中 generation 10 的 `h_0097` 是关键节点：

```text
distance_update_policy.type = utility_ratio
strength = 0.3
cap_multiplier = 20.0
objective = 0.07947
```

这说明 `utility_ratio` 家族是本次搜索中非常重要的方向。

#### 阶段 3：关键跃迁（generation 10 -> 11）

```text
gen10 best = 0.07947
gen11 best = 0.00479
```

关键候选 `h_0107` 加入：

```text
schedule_policy.strength_schedule = stagnation_boost
early_iterations = 100
stagnation_multiplier = 2.0
```

这一步把训练 objective 从 `0.07947` 大幅降到 `0.00479`。

这非常重要，因为它说明：

```text
不是单个 penalty 或 distance update 就能达到最好结果，
而是 distance update + penalty decay + escape + schedule 的组合产生了强效果。
```

#### 阶段 4：重启策略和 penalty 策略微调（generation 14 -> 20）

关键节点：

```text
h_0140: restart_policy = decay_penalty, objective = 0.00456
h_0157: penalty_policy.edge_selection = tour_utility, top_k = 15, objective = 0.00916
h_0183: penalty_policy.top_k = 10, objective = 0.00437
```

最终训练最优候选：

```text
h_0183 / h_0192 / h_0193 / h_0198
training objective = 0.00437
```

---

## 7. 最终 ARCH_SPEC 结构是什么？

最终种群中训练最优的一组结构高度相似，核心为：

```text
distance_update_policy.type = utility_ratio
penalty_policy.edge_selection = tour_utility or top_gap
penalty_policy.decay = 0.1
escape_policy.moves = 3
restart_policy.action = decay_penalty
restart_policy.trigger = periodic
restart_policy.period = 50
schedule_policy.strength_schedule = stagnation_boost
schedule_policy.stagnation_multiplier = 2.0
```

其中训练最优之一 `h_0183`：

```text
training objective = 0.00437

distance_update_policy:
  type = utility_ratio
  strength = 0.3
  usage_power = 1.0
  length_weight = 0.0
  cap_multiplier = 20.0

penalty_policy:
  edge_selection = tour_utility
  top_k = 10
  increment = 1.0
  decay = 0.1

escape_policy:
  trigger = stagnation
  moves = 3
  period = 40
  stagnation_threshold = 50
  size_threshold = 150
  max_moves = 3

restart_policy:
  action = decay_penalty
  trigger = periodic
  period = 50
  stagnation_threshold = 30
  penalty_decay = 0.05

schedule_policy:
  strength_schedule = stagnation_boost
  early_iterations = 100
  late_multiplier = 1.0
  stagnation_multiplier = 2.0
  size_threshold = 150
  size_multiplier = 1.0
```

---

## 8. 从结果看，哪些组件有效？

### 8.1 ARCH_SPEC 结构化表示：有效

证据：

1. 20 代都能正常完成；
2. 每个候选都有明确 parent、delta、changed layer；
3. 候选代码由 spec 确定性生成，减少随机代码错误；
4. 最终找到的是多层 GLS 组合，而不是单一函数微调。

如果没有 ARCH_SPEC，我们很难说清楚最终算法为什么有效，也很难向导师解释每个改动的贡献。

当前结果说明：

```text
结构化搜索空间是有效的。
```

---

### 8.2 单层受控实验：有效

最终最优链条可以清楚追踪：

```text
spec_linear_top8_periodic_escape_seed
  -> h_0009: escape moves=3
  -> h_0047: distance strength=0.6
  -> h_0053: penalty decay=0.1
  -> h_0071: escape trigger=stagnation
  -> h_0097: distance type=utility_ratio
  -> h_0107: schedule=stagnation_boost
  -> h_0140: restart=decay_penalty
  -> h_0157: penalty edge_selection=tour_utility, top_k=15
  -> h_0183: penalty top_k=10
```

这个链条非常像人类专家逐步调试算法：

1. 先调扰动强度；
2. 再加 penalty decay；
3. 再换 distance update 信号；
4. 再加入停滞增强调度；
5. 再加入 penalty decay restart；
6. 最后细调 penalty 选边 top-k。

这说明框架不只是随机采样，而是在逐步构造一个复杂机制组合。

---

### 8.3 generation planner：有效，但仍有改进空间

有效证据：

```text
generation_plan 成功 20 次
generation_plan_failed = 0
仅 generation 18 有一次 retry，最后成功
```

planner 能够持续输出合法计划，并且计划中的 `child_spec` 大多数能通过单层变更检查。

不足：

后期出现了较多重复 spec：

```text
Duplicate ARCH_SPEC code generated = 16
```

这说明 planner 已经能围绕有效方向做局部搜索，但对“该方向是否已饱和”的判断还不够主动。人类专家看到多个微调无效后，可能会转向其他机制家族；当前 planner 后期仍倾向于在 `utility_ratio` 附近反复微调。

---

### 8.4 generation reflection / expert notebook：有效，但有噪声

reflection 每代都成功，并且 expert notebook 中确实记录了有用经验，例如：

```text
penalty_policy.top_k 从 15 降到 10 在 tour_utility 下改善 tail_gap
restart_policy decay_penalty 比 reset_to_initial 更稳定
escape_policy moves=1 会造成 single_extreme_training_failure
```

这些总结与最终结构基本一致。

但 notebook 中也存在一些看起来矛盾的总结，例如某些 `stagnation_boost` 同时被描述为有效和可能过强。这是因为不同父代、不同组合下同一机制的作用不同，reflection 目前还没有很好地区分“局部上下文依赖”。

因此结论是：

```text
reflection 有效，但机制归因还需要更精细的上下文条件。
```

---

### 8.5 quality-diversity survival：部分有效

最终 TSPLib 最优个体不是训练最优个体：

| candidate | training objective | TSPLib average gap |
|---|---:|---:|
| h_0183 | 0.00437 | 0.22943 |
| h_0192 | 0.00437 | 0.22943 |
| h_0193 | 0.00437 | 0.22943 |
| h_0198 | 0.00437 | 0.22943 |
| h_0157 | 0.00916 | 0.21278 |

`h_0157` 训练分数不是最好，但在测试集上最好。它能保留到最终种群，说明 quality-diversity 没有把所有非训练最优个体都淘汰掉。

这说明 quality-diversity 有价值。

但不足是：最终 10 个个体全部集中在 `utility_ratio_penalty` 家族。也就是说，它保留的是同一大机制家族内部的小差异，而不是多个机制家族级别的备选路线。

后续应增强：

```text
mechanism-family-level diversity
```

---

### 8.6 no fallback / fail closed：有效

本次没有 planner 失败后用默认计划替代，也没有候选失败后自动修补成别的机制。

这保证了实验结论可靠：

```text
所有进入结果的候选都是真实经过 planner -> spec audit -> evaluator 的候选。
```

这对论文非常重要，因为可以避免实验污染。

---

## 9. 哪些组件目前效果不足？

### 9.1 prompt compression 效果很弱

本次 cost audit 汇总：

```text
prompt_tokens = 576944
completion_tokens = 118994
total_tokens = 695938
cache_hit_tokens = 28800
cache_miss_tokens = 548144
cache_hit_rate = 4.99%
compression_saved_chars = 17571
raw_chars = 1857468
compression_saved_ratio = 0.95%
```

这说明当前 prompt compression 基本没有显著节省成本。

原因是当前 prompt 结构中大量动态内容每代变化，包括：

```text
population summary
training diagnostics
notebook
mechanism archive
recent events
compact context
```

即使压缩 leaf value，整体 prompt 的稳定前缀仍然不够长，导致 cache 命中率也只有约 5%。

结论：

```text
compression / cache 目前不是有效组件，只是成本审计组件。
```

后续应该重新设计 prompt 结构，而不是只做字符级压缩。

---

### 9.2 duplicate spec 抑制需要前置

当前 duplicate 检查发生在候选生成后：

```text
planner 输出 child_spec -> 系统生成代码 -> 发现重复 -> 拒绝
```

这保护了实验纯净性，但浪费了 offspring slot。

本次有：

```text
16 / 200 = 8%
```

的计划因为重复 spec 被拒绝。

后续可以在 planner 输入中显式加入：

```text
forbidden_specs
forbidden_niches
recent_duplicate_deltas
```

让 planner 在计划阶段避免重复。

---

### 9.3 训练目标与测试泛化仍有错位

训练最优并不是 TSPLib 最优：

```text
training best: h_0183 / h_0192 / h_0193 / h_0198, objective=0.00437, TSPLib=0.22943
TSPLib best: h_0157, training objective=0.00916, TSPLib=0.21278
```

这说明训练平均 gap 不能完全代表泛化能力。

具体看 TSPLib，当前最优 `h_0157` 仍有一些困难实例：

```text
eil101.tsp: 1.78245
eil76.tsp: 1.18384
rat195.tsp: 0.84109
rat99.tsp: 0.68074
eil51.tsp: 0.67412
```

因此下一步需要更强的泛化诊断，例如：

```text
训练实例聚类
tail-risk objective
held-out probe，但要避免污染主测试集
按实例类型的 failure signature
```

---

## 10. TSPLib 评估结果

### 10.1 我们框架最终 population_generation_20

评估文件：

```text
eohagent/results/tsp_gls_arch_spec_deepseek_10x20_v1/population_generation_20_tsplib_eval.csv
```

汇总结果：

| metric | value |
|---|---:|
| candidates evaluated | 10 |
| best TSPLib average gap | 0.21278 |
| median TSPLib average gap | 0.23433 |
| worst TSPLib average gap | 0.23923 |
| best training objective | 0.00437 |
| median training objective | 0.00456 |
| worst training objective | 0.00916 |

前 5 个 TSPLib 候选：

| candidate | training objective | TSPLib average gap |
|---|---:|---:|
| h_0157 | 0.00916 | 0.21278 |
| h_0183 | 0.00437 | 0.22943 |
| h_0192 | 0.00437 | 0.22943 |
| h_0193 | 0.00437 | 0.22943 |
| h_0198 | 0.00437 | 0.22943 |

### 10.2 与之前单函数版本的参考比较

之前部分单函数/机制版本结果：

| method/version | best TSPLib average gap | median TSPLib gap |
|---|---:|---:|
| fixed-function/mech v3 | 0.27147 | 0.29832 |
| fixed-function/mech v5 | 0.26615 | 0.93654 |
| current ARCH_SPEC v1 | 0.21278 | 0.23433 |

说明当前 `ARCH_SPEC` 版本不仅训练分数更好，在 TSPLib 上也比之前单函数搜索版本更好。

这支持我们的核心创新：

```text
从单函数启发式搜索升级到受约束的 GLS 架构级设计是有效的。
```

---

## 11. ReEVO 对比实验

### 11.1 ReEVO 是什么

ReEVO 论文标题是：

```text
ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution
NeurIPS 2024
```

ReEVO 的核心流程是：

```text
population initialization
selection
short-term reflection
crossover
long-term reflection
elitist mutation
```

它的定位是 LLM-based hyper-heuristic，即用 LLM 生成启发式代码，并通过反思和进化搜索不断改进。

这与我们的方向非常相关，所以它是一个有代表性的 baseline。

---

### 11.2 当前 ReEVO 对比实验配置

本次 ReEVO 不是直接运行论文原始脚本，而是通过 LLM4AD 平台适配到我们的 TSP-GLS evaluator。

运行配置：

```json
{
  "method": "reevo",
  "model": "deepseek-v4-flash",
  "api_host": "api.deepseek.com",
  "temperature": 0.7,
  "max_samples": 200,
  "max_generations": 20,
  "pop_size": 10,
  "num_samplers": 5,
  "num_evaluators": 5,
  "eval_timeout": 500,
  "n_inst_eva": 64,
  "time_limit": 60,
  "ite_max": 1000,
  "perturbation_moves": 1,
  "example_dir": "examples/user_tsp_gls_OPRO",
  "seed_file": null,
  "export_top_k": 10
}
```

实际 profiler 中记录了 206 个 valid scores。原因是并行采样/评估可能导致到达 200 附近时略微 overshoot。

---

### 11.3 ReEVO 对比结果

评估文件：

```text
eohagent/results/baselines/reevo_deepseek_10x200_v1/population_final_tsplib_eval.csv
```

汇总结果：

| metric | value |
|---|---:|
| candidates exported/evaluated | 10 |
| best TSPLib average gap | 0.34209 |
| median TSPLib average gap | 0.35399 |
| worst TSPLib average gap | 0.35399 |
| best training objective among exported candidates | 0.02838 |
| median training objective | 0.02838 |
| worst training objective | 0.02942 |

前几个候选：

| candidate | training objective | TSPLib average gap |
|---|---:|---:|
| reevo_0008 | 0.02942 | 0.34209 |
| reevo_0009 | 0.02942 | 0.34209 |
| reevo_0010 | 0.02942 | 0.34209 |
| reevo_0001 | 0.02838 | 0.35399 |
| reevo_0002 | 0.02838 | 0.35399 |

### 11.4 我们 vs ReEVO

| method | training best | TSPLib best | TSPLib median |
|---|---:|---:|---:|
| ARCH_SPEC framework | 0.00437 | 0.21278 | 0.23433 |
| ReEVO adapted baseline | 0.02838 | 0.34209 | 0.35399 |

相对提升：

```text
TSPLib best gap: 0.34209 -> 0.21278
absolute improvement = 0.12931
relative improvement ≈ 37.8%
```

也就是说，在当前适配设置下，我们的结果明显优于 ReEVO baseline。

---

## 12. 为什么 ReEVO 当前对比不是论文原始参数？

这个点必须在汇报时讲清楚，否则导师可能会问：

```text
你这个 ReEVO 跑法是不是公平？是不是复现论文？
```

答案是：

```text
这次 ReEVO 是为了和我们框架公平比较而适配的 baseline，不是 ReEVO 论文原始实验设置。
```

### 12.1 ReEVO 论文默认参数

ReEVO 论文 Appendix C Table 8 给出的默认参数包括：

| parameter | paper value |
|---|---:|
| LLM | gpt-3.5-turbo |
| LLM temperature | 1 |
| population size | 10 |
| number of initial generation | 30 |
| maximum number of evaluations | 100 |
| crossover rate | 1 |
| mutation rate | 0.5 |

此外，论文中的 TSP-GLS 是 KGLS-ReEvo 设置，GLS evaluation 参数和我们当前 OPRO GLS evaluator 不完全一致。论文 Table 9 中不同 TSP size 使用不同 perturbation moves 和 iteration 设置，例如 TSP100/TSP200 的 perturbation moves 是 40，iteration 数也不是我们当前统一的 `ite_max=1000`。

### 12.2 我们这次为什么改参数？

我们当前的实验目标不是复现 ReEVO 论文的 KGLS-ReEvo 结果，而是回答：

```text
在同一个 TSP-GLS 训练集、同一个 evaluator、同一个 DeepSeek API 下，
传统 LLM4AD/ReEVO 单函数搜索 baseline 和我们的 ARCH_SPEC 架构级框架相比如何？
```

因此我们刻意对齐了以下内容：

| item | aligned setting |
|---|---|
| LLM | deepseek-v4-flash |
| API host | api.deepseek.com |
| training set | examples/user_tsp_gls_OPRO, 64 instances |
| objective | average training gap |
| evaluator timeout | 500 s |
| time_limit | 60 |
| ite_max | 1000 |
| perturbation_moves | 1 |
| population size | 10 |
| exported candidates | top 10 |

这些对齐是必要的，否则差异可能来自 evaluator、训练集、时间限制或 perturbation 参数，而不是搜索框架本身。

### 12.3 ReEVO 当前设置与论文不同在哪里？

| item | ReEVO paper | our adapted ReEVO |
|---|---|---|
| LLM | gpt-3.5-turbo | deepseek-v4-flash |
| temperature | 1 | 0.7 |
| maximum evaluations | 100 | 200 budget，实际 206 scores |
| TSP-GLS implementation | KGLS-ReEvo / paper GLS setting | our OPRO-aligned TSP-GLS evaluator |
| perturbation moves | size-dependent, e.g. 40 for TSP100/TSP200 in paper table | 1 |
| evaluator objective | paper-specific GLS performance | our 64-instance mean gap |
| final testing | paper held-out setup | our TSPLib evaluator |
| initialization | ReEVO paper setting with optional seeds/task specification | LLM4AD ReEVO default, no our expert ARCH_SPEC seeds |

因此，这个 baseline 的含义是：

```text
ReEVO-style reflective evolutionary single-function search under our current evaluation protocol。
```

而不是：

```text
ReEVO paper full reproduction。
```

这点在 PPT 中要明确写出。

---

## 13. 为什么我们当前结果比 ReEVO 好？

可能原因有几类。

### 13.1 搜索空间不同：这是我们的核心创新

ReEVO baseline 搜索的是单函数/代码级启发式，主要围绕 `update_edge_distance` 或对应的 heuristic function 改写。

我们的框架搜索的是 GLS 架构级组合：

```text
distance update
penalty edge selection
escape schedule
restart policy
strength schedule
```

这不是要和 ReEVO 对齐的部分，因为这正是我们的创新点。

当前结果显示，最终有效结构确实需要多层组合，例如：

```text
utility_ratio distance update
+ penalty decay
+ stagnation/periodic escape
+ decay_penalty restart
+ stagnation_boost schedule
```

如果 baseline 只能改一个函数，就很难同时探索这些结构组合。

### 13.2 结构化约束减少无效代码搜索

我们的 LLM 不直接写任意代码，而是写结构化 child_spec。系统确定性生成代码。

这降低了：

```text
语法错误
接口错误
无意义代码变化
不可审计大改
```

ReEVO 则更接近开放代码搜索，灵活性更高，但在当前任务和预算下可能更难高效落到有效 GLS 结构。

### 13.3 专家种子初始化有效

我们的初始化包含一组手工设计的 ARCH_SPEC seeds。它们覆盖多种机制家族，例如：

```text
linear_usage
length_weighted_usage
utility_ratio
rank_usage
smooth_saturation
periodic escape
stagnation restart
```

ReEVO 当前 baseline 没有使用我们的 ARCH_SPEC expert seeds。这不是 bug，而是因为 expert seed initialization 本身是我们框架的一部分。

因此可以这样解释：

```text
我们不是仅靠 LLM 从零搜索，而是把专家先验结构化为可进化 seeds，
然后让 LLM 做受控实验设计。
```

这更接近真实专家工作流。

### 13.4 训练诊断和 reflection 更贴近算法机制

我们的 reflection 不只是比较两个代码好坏，而是记录：

```text
which layer changed
which delta helped
tail gap changed how
failure signature changed how
```

因此下一代 planner 能看到机制层面的反馈，而不是只看到整体分数。

---

## 14. 当前框架是否已经“完美模仿人类专家”？

结论：

```text
还没有完美实现，但已经明显接近。
```

### 14.1 已经实现的专家行为

当前框架已经做到：

1. 从专家 seed 出发；
2. 每代先做实验计划；
3. 每个候选都有机制假设；
4. 每个候选只改变一个架构层；
5. 失败候选不 fallback；
6. 每代做结果反思；
7. 下一代利用 notebook 和 archive；
8. 最终能组合出多层有效 GLS 架构。

这些已经非常接近专家工作流。

### 14.2 还不像专家的地方

#### 1. 对“方向饱和”的判断不够主动

后期大量围绕 `utility_ratio` 微调，重复 spec 增多。人类专家可能会更早意识到这个方向已经接近饱和，转而探索其他机制家族。

#### 2. 机制家族级 diversity 不够

最终种群全部集中在 `utility_ratio_penalty` 家族。人类专家通常会保留几条备选路线，以防训练最优路线泛化不好。

#### 3. 泛化风险判断不够强

训练最优不是 TSPLib 最优，说明框架还不能完全识别泛化风险。

#### 4. 成本管理较弱

prompt compression 和 cache hit 当前贡献很小。这不影响算法质量，但会影响大规模实验成本。

---

## 15. PPT 可以怎么讲？

### Slide 1: 问题背景

```text
目标：让 LLM 自动设计 TSP-GLS 启发式算法
问题：直接生成代码不可控、不可解释、难以归因
我们的思路：把 GLS 算法设计转成结构化 ARCH_SPEC 搜索
```

### Slide 2: 框架总览图

建议画流程图：

```text
Expert Seeds
   ↓
Generation Planner
   ↓
Controlled ARCH_SPEC Plans
   ↓
Single-layer Audit
   ↓
Deterministic Code Generation
   ↓
Formal Evaluator
   ↓
Training Diagnostics
   ↓
Quality-Diversity Selection
   ↓
Generation Reflection / Expert Notebook
   ↓
Next Generation
```

### Slide 3: ARCH_SPEC 五层结构

展示五个开放层：

```text
distance_update_policy
penalty_policy
escape_policy
restart_policy
schedule_policy
```

强调：每个候选只允许改一层。

### Slide 4: 训练曲线

展示 best objective 从：

```text
0.18540 -> 0.15471 -> 0.07947 -> 0.00479 -> 0.00437
```

强调关键跃迁在 generation 11，来自 `stagnation_boost schedule`。

### Slide 5: 最终算法结构

展示最终结构：

```text
utility_ratio distance update
penalty decay = 0.1
tour_utility / top_gap edge selection
escape moves = 3
restart = decay_penalty
schedule = stagnation_boost
```

说明这是多层组合，不是单函数微调。

### Slide 6: TSPLib 结果

表格：

| method | TSPLib best | TSPLib median |
|---|---:|---:|
| ARCH_SPEC | 0.21278 | 0.23433 |
| ReEVO adapted | 0.34209 | 0.35399 |

### Slide 7: ReEVO 对比说明

重点写：

```text
ReEVO 是强相关 baseline，但本次不是论文原始参数复现。
为了公平比较，我们对齐了 DeepSeek API、训练集、evaluator、time limit、perturbation_moves。
因此结果表示：在我们当前 evaluator 下，ReEVO-style single-function search 不如 ARCH_SPEC architecture-level search。
```

### Slide 8: 组件有效性总结

| component | status | evidence |
|---|---|---|
| ARCH_SPEC | 有效 | 训练和 TSPLib 都提升 |
| 单层审计 | 有效 | 可归因链条清晰 |
| generation planner | 有效但需改进 | 20 代成功，但重复 spec |
| reflection | 有效但有噪声 | notebook 记录有用经验 |
| quality-diversity | 部分有效 | 保住 TSPLib 最优 h_0157 |
| prompt compression/cache | 暂不有效 | cache hit 约 5%，compression saved 约 0.95% |

### Slide 9: 下一步工作

```text
1. 加强机制家族级 diversity
2. 前置 duplicate spec 抑制
3. 改进泛化风险诊断
4. 重构 prompt/cache 结构降低成本
5. 继续补充更严格 baseline，例如 EoH/ReEVO paper-like v2
```

---

## 16. 最终结论

当前阶段最重要的结论是：

```text
ARCH_SPEC 架构级搜索路线是有效的。
```

它不仅在训练集上明显降低 gap，而且在 TSPLib 测试集上优于之前单函数版本和当前 ReEVO 适配 baseline。

本框架的关键价值不是“多跑了几个候选”，而是把 LLM 自动算法设计过程变成了：

```text
结构化、可审计、可归因、可反思的专家式实验循环。
```

当前还需要改进的主要问题是：

```text
机制家族多样性不足
泛化风险识别不足
重复 spec 抑制不足
prompt/cache 成本优化不足
```

但作为阶段性结果，这已经足以支持向导师汇报：

```text
我们的方法不仅能运行，而且已经展现出比 ReEVO-style baseline 更好的效果；
更重要的是，它提供了一个更接近人类专家设计算法过程的自动算法设计框架。
```

---

# 附录 A：当前框架工作流详细版（给不熟悉代码的人看的逐步说明）

这一节专门解释：**我们的框架一次完整实验到底是怎么运行的**。  
可以把它理解为“从配置文件开始，到最终得到 20 代种群结果”的完整流水线。

---

## A.1 总体一句话

当前框架不是让 LLM 直接随便写代码，而是让 LLM 先做“实验设计”，输出一个结构化的 GLS 架构修改方案，然后系统把这个方案确定性转成代码，再用正式 evaluator 评估，最后根据结果做反思并进入下一代。

最核心的流程是：

```text
读取配置
  -> 评估专家种子
  -> 进入第 1 代
      -> 选择父代
      -> LLM 生成这一代的实验计划
      -> 检查每个计划是否合法
      -> 根据 child_spec 确定性生成候选代码
      -> contract check
      -> 正式 evaluator 评估
      -> 记录训练诊断
      -> 更新种群
      -> LLM 做 generation reflection
      -> 写入 notebook / archive / compact context
  -> 重复到第 20 代
  -> 导出最终 population
  -> 额外用 TSPLib 脚本做测试集评估
```

---

## A.2 运行入口：配置文件驱动整个实验

这次主实验配置文件是：

```text
eohagent/config/tsp_gls_arch_spec_deepseek_10x20.yaml
```

核心配置包括：

```yaml
project.problem: tsp_gls_arch_spec

evaluator:
  function_name: CandidateGLSArchitecture
  architecture_class_name: CandidateGLSArchitecture
  n_inst_eva: 64
  time_limit: 60
  ite_max: 1000
  perturbation_moves: 1
  timeout: 500

evolution:
  pop_size: 10
  generations: 20
  offspring_per_generation: 10
  n_proc: 5
  survival_selection: quality_diversity

architecture.open_layers:
  - distance_update_policy
  - penalty_policy
  - escape_policy
  - restart_policy
  - schedule_policy
```

这个配置决定了：

1. 做什么问题：`tsp_gls_arch_spec`；
2. 每个候选必须实现什么类：`CandidateGLSArchitecture`；
3. 每代多少候选：10 个；
4. 总共多少代：20 代；
5. 每次 formal evaluation 的限制：64 个训练实例、每实例 60 秒、最多 1000 次迭代、扰动步数为 1；
6. 搜索空间有 5 个可修改的架构层。

---

## A.3 引擎初始化：ExpertEoHEngine 做总调度

代码中的总调度类是：

```text
eohagent/core/engine.py
class ExpertEoHEngine
```

它负责把所有组件连起来。

初始化时会做这些事情：

```text
1. 读取 config
2. 检查 workflow contract
3. 准备 output_dir
4. 构建 evaluator
5. 构建 memory manager
6. 构建 population manager
7. 构建 LLM client
8. 构建 JSONCall
9. 构建 AgentHarness
10. 构建 GenerationPlanner
```

可以理解为：

```text
ExpertEoHEngine = 整个实验的导演
GenerationPlanner = 负责让 LLM 设计每代实验
MemoryManager = 负责记忆过去的实验经验
PopulationManager = 负责选择谁活下来
EvaluatorBridge = 负责真正跑 TSP-GLS
AgentHarness / ToolRouter = 负责候选生成和检查
```

---

## A.4 workflow contract：先保证实验不会跑偏

在真正开始之前，引擎会检查当前配置是不是符合 ARCH_SPEC 工作流要求。

对于 `tsp_gls_arch_spec`，它要求：

```text
architecture_class_name 必须是 CandidateGLSArchitecture
open_layers 必须正好是固定 5 层
survival_selection 必须是 quality_diversity
```

如果不满足，直接报错，不会继续跑。

这一步的意义是：

```text
防止我们不小心用错 evaluator、用错搜索空间、或者变成普通 EoH 单函数搜索。
```

这也是实验纯净性的一部分。

---

## A.5 第 0 代：bootstrap 专家种子

实验开始后不是先问 LLM，而是先评估专家种子。

配置：

```yaml
bootstrap.use_baseline_seed: true
bootstrap.seed_path: ./eohagent/seeds/tsp_gls_arch_spec_baseline.json
```

这个 seed 文件里有 10 个手写的 ARCH_SPEC 初始算法，例如：

```text
linear_usage + top8 + standard
length_weighted_usage + top8
utility_ratio + top8
rank_usage + top8
smooth_saturation + top8
linear_usage + top3
linear_usage + top12
linear_usage + decay
linear_usage + periodic escape
linear_usage + stagnation restart
```

系统会并行评估这 10 个 seed，得到第 0 代 population。

本次第 0 代最好的 seed 是：

```text
spec_linear_top12_standard_seed
objective = 0.18540
```

注意：这个步骤很重要。  
它等价于人类专家先给出一组初始设计方案，而不是让 LLM 从完全空白开始。

---

## A.6 每一代开始：先选择父代 parent

每代要生成 10 个 offspring。

在生成计划前，系统会先从当前 population 中选出 10 个 parent slot。

当前配置：

```yaml
parent_selection: rank
allow_archive_parents: false
```

含义是：

```text
只从当前 population 中选父代，不从历史 archive 中选。
父代选择偏向 objective 更好的个体。
```

然后这 10 个 parent 会被写进 LLM 的 planner prompt 里。

LLM 不是自由选择任意 parent，而是必须按照系统给的 slot 来写计划。

---

## A.7 GenerationPlanner：LLM 先写“实验计划”，不是直接写代码

每一代最关键的一步是：

```text
GenerationPlanner.plan(...)
```

它会构造一个 JSON prompt，告诉 LLM：

```text
你现在要设计一代 GLS ARCH_SPEC 受控实验。
你必须返回 10 个 offspring_plans。
每个 offspring_plan 必须对应系统指定的 parent。
每个 child_spec 必须只改变一个架构层。
controlled_component 必须是 5 个开放层之一。
```

LLM 需要返回类似下面的结构：

```json
{
  "generation_goal": "Test penalty top_k and tour_utility variants around the best utility_ratio family",
  "avoid_repeating": ["avoid moves=1 because it caused extreme failures"],
  "offspring_plans": [
    {
      "offspring_index": 1,
      "parent_id": "h_0157",
      "group_id": "g_penalty_topk",
      "variant_id": "v_top10",
      "group_hypothesis": "Narrowing top_k may reduce tail failures",
      "controlled_component": "penalty_policy",
      "delta": "Set penalty_policy.top_k to 10",
      "preserve": "Keep distance, escape, restart, schedule unchanged",
      "expected_effect": "Lower mean/tail gap without introducing extreme failure",
      "child_spec": {
        "distance_update_policy": {...},
        "penalty_policy": {...},
        "escape_policy": {...},
        "restart_policy": {...},
        "schedule_policy": {...}
      }
    }
  ]
}
```

这里最重要的是：

```text
LLM 输出的是 child_spec，不是 Python 代码。
```

也就是说 LLM 是在做“算法架构设计”，不是直接写函数。

---

## A.8 planner 输出后，系统做严格合法性检查

LLM 返回 JSON 后，系统不会直接相信它。

`GenerationPlanner._normalize_plans` 会逐个检查：

```text
1. offspring_plans 数量必须等于 10
2. offspring_index 必须严格匹配 slot
3. parent_id 必须等于系统指定 parent
4. controlled_component 必须非空
5. controlled_component 必须属于 5 个 ARCH_SPEC 层
6. child_spec 必须存在
7. child_spec 必须是合法 ARCH_SPEC
8. child_spec 相比 parent 只能改变 controlled_component 这一层
```

第 8 点最重要。  
例如，如果 `controlled_component = penalty_policy`，那么：

```text
parent.distance_update_policy 必须等于 child.distance_update_policy
parent.escape_policy 必须等于 child.escape_policy
parent.restart_policy 必须等于 child.restart_policy
parent.schedule_policy 必须等于 child.schedule_policy
只有 parent.penalty_policy 可以不同
```

如果不满足，整个 planner attempt 会失败并 retry。

这就是“受控实验”的实现方式。

---

## A.9 child_spec 如何变成候选代码？

通过检查后，每个 offspring plan 会进入：

```text
ExpertEoHEngine._generate_one_arch_spec_offspring(...)
```

这个函数不会再问 LLM 写代码，而是调用：

```text
_candidate_from_arch_spec_plan(...)
```

它做几件事：

```text
1. 从 parent 中读出 parent_arch_spec
2. 读取 plan 中的 child_spec
3. 再次验证只改了 controlled_component 一层
4. 调用 build_arch_spec_code(child_spec)
5. 构造 Candidate 对象
```

生成的代码大概是：

```python
from eohagent.problems.tsp_gls_arch_spec.spec_runtime import SpecCandidateGLSArchitecture

ARCH_SPEC = {
  ...
}

class CandidateGLSArchitecture(SpecCandidateGLSArchitecture):
    def __init__(self):
        super().__init__(ARCH_SPEC)
```

也就是说，候选代码是系统模板 + LLM 设计的 ARCH_SPEC。

LLM 没有机会直接写任意复杂 Python 逻辑。

---

## A.10 SpecCandidateGLSArchitecture：真正执行 GLS 逻辑的地方

所有候选最终都继承：

```text
eohagent/problems/tsp_gls_arch_spec/spec_runtime.py
class SpecCandidateGLSArchitecture
```

这个 runtime 类根据 ARCH_SPEC 实现真正的 GLS 行为。

它包含这些方法：

```text
update_edge_distance(...)
select_penalty_edges(...)
penalty_increment(...)
penalty_decay(...)
perturbation_schedule(...)
restart_policy(...)
restart_action(...)
restart_penalty_decay(...)
```

所以候选 spec 的不同，会影响这些具体行为。

例如：

### 如果 distance_update_policy.type = utility_ratio

那么边距离更新时，信号会更关注：

```text
edge length / (1 + usage)
```

也就是高 utility 的边会被更重点处理。

### 如果 penalty_policy.edge_selection = tour_utility

那么选择加 penalty 的边时，会按：

```text
edge_weight / (1 + edge_penalty)
```

来给当前 tour 上的边排序。

### 如果 escape_policy.moves = 3

那么扰动时会使用 3 次 move，而不是默认 1 次。

### 如果 restart_policy.action = decay_penalty

那么触发 restart 时，不是简单 reset 到 best，而是衰减 penalty，让搜索从过度惩罚中恢复。

### 如果 schedule_policy.strength_schedule = stagnation_boost

那么搜索停滞时，distance update 的强度会被放大。

这就是为什么 ARCH_SPEC 是“架构级设计”：它影响 GLS 的多个组件。

---

## A.11 contract check：候选进 evaluator 前再检查

候选代码生成后，会进入 contract check：

```text
self.harness.tool_router._check_code(candidate.code)
```

这一步检查：

```text
Python 语法是否能解析
是否包含需要的类/函数结构
是否有明显接口问题
```

如果 contract check 失败，该候选直接拒绝，不进入 evaluator。

---

## A.12 duplicate check：重复架构直接拒绝

候选进入评估前，还会检查它的代码是否和当前 population 里的代码重复。

如果重复，会记录：

```text
Duplicate ARCH_SPEC code generated
```

然后拒绝该 offspring。

本次 20 代中有 16 个重复 spec 被拒绝。

这不是系统错误，而是说明 planner 有时提出了已经存在的架构。

当前做法是：

```text
重复候选不评估，不进入种群。
```

这样可以防止结果污染。

---

## A.13 formal evaluation：真正跑 TSP-GLS

通过 contract check 和 duplicate check 后，候选进入正式评估：

```text
_evaluate_individual(candidate.code, candidate.id)
```

这个函数会：

```text
1. ast.parse 检查代码语法
2. 新建 ThreadPoolExecutor
3. 调用 evaluator.evaluate(code)
4. 设置 timeout=500 秒
5. 得到 fitness
6. 转成 objective
```

正式 evaluator 会把候选代码加载进去，然后在训练集上跑 TSP-GLS。

本次训练 evaluator 设置是：

```text
64 个训练实例
每个实例 time_limit = 60
ite_max = 1000
perturbation_moves = 1
```

最终返回：

```text
64 个训练实例平均 gap
```

越小越好。

---

## A.14 evaluation evidence：不仅记录平均分，还记录诊断信息

评估完成后，系统会把 objective 写到 candidate 上，同时调用：

```text
_attach_evaluation_evidence(candidate, parent)
```

它会记录：

```text
formal_evaluator:
  mean_gap
  std_gap
  max_gap
  min_gap
  per_instance_gap
  worst_instances
  best_instances

training_diagnosis:
  mean_gap
  std_gap
  tail_gap
  max_gap
  failure_signature
  worst_training_instances
  improved_instances_vs_parent
  worsened_instances_vs_parent
  mean_delta_vs_parent
  tail_delta_vs_parent
```

这些信息非常重要。  
因为下一代 planner 不只知道谁平均分低，还知道：

```text
它是不是尾部很差？
是不是单个实例崩了？
相对父代改善了哪些实例？
失败模式是什么？
```

这就是“专家诊断”的数据来源。

---

## A.15 候选进入 memory

每个成功评估的候选会写入多个地方：

```text
candidate_store.json
session_log.jsonl
mechanism_archive.json
expert_notebook.json
progress.jsonl
```

大概含义是：

| 文件 | 作用 |
|---|---|
| candidate_store.json | 存所有候选完整信息 |
| session_log.jsonl | 存每一步事件日志 |
| mechanism_archive.json | 按机制家族整理候选经验 |
| expert_notebook.json | 存 LLM 反思后的专家经验 |
| progress.jsonl | 存方便查看训练进度的事件 |

---

## A.16 population update：谁活到下一代？

每代 10 个 offspring 评估完后，系统把：

```text
old population + evaluated offspring
```

合并，然后交给：

```text
PopulationManager.update(...)
```

当前使用：

```yaml
survival_selection: quality_diversity
pop_size: 10
```

它不是简单只选平均 objective 最小的 10 个，而是先看 Pareto-like 指标：

```text
objective
tail_gap
max_gap
std_gap
```

再结合 ARCH_SPEC niche 做保留。

这样做的目的：

```text
既保留好分数，也尽量保留不同结构，避免全是同一个候选的微小重复。
```

不过本次实验也暴露了问题：最终还是集中在 `utility_ratio` 家族，说明 mechanism-family-level diversity 还需要增强。

---

## A.17 generation reflection：每代结束后 LLM 总结经验

种群更新后，如果这一代有成功 evaluated offspring，系统会调用：

```text
GenerationPlanner.reflect(...)
```

这一步让 LLM 总结本代实验。

输入包括：

```text
surviving_population
evaluated_offspring
controlled_experiment_results
recent formal eval events
training diagnostics
```

LLM 输出：

```json
{
  "validated_principles": [...],
  "failed_hypotheses": [...],
  "supported_deltas": [...],
  "refuted_deltas": [...],
  "ambiguous_deltas": [...],
  "failure_patterns": [...],
  "family_status": [...],
  "next_generation_priorities": [...],
  "open_questions": [...]
}
```

这些内容会合并进 expert notebook。

下一代 planner 会看到这些 notebook 内容，从而避免重复失败方向，或者继续深挖有效方向。

---

## A.18 context compaction：控制上下文长度

因为每代都有 population、diagnostics、notebook、archive，prompt 会越来越长。

所以系统有 memory compaction：

```text
compact_context.json
```

它会保留：

```text
current_best
recent_successes
recent_failures
next_focus
mechanism_cards
delta_cards
generation_attributions
working_memory
```

这样 planner 不需要每次看到所有历史细节，只看到压缩后的重点。

不过本次实验显示：

```text
prompt compression / cache 目前成本收益不大
```

这属于成本优化问题，不影响主算法逻辑。

---

## A.19 一代实验的具体例子

以最终强链条中的一个关键变化为例。

假设当前 parent 是：

```text
h_0097
objective = 0.07947
核心结构：utility_ratio distance update + penalty decay + stagnation escape
```

planner 在 generation 11 输出一个 plan：

```text
controlled_component = schedule_policy
delta = Set schedule_policy.strength_schedule to stagnation_boost
```

child_spec 只有 schedule_policy 变化：

```text
before:
  schedule_policy.strength_schedule = constant

after:
  schedule_policy.strength_schedule = stagnation_boost
  early_iterations = 100
  stagnation_multiplier = 2.0
```

系统检查：

```text
changed_layers == [schedule_policy]
```

通过。

然后生成候选：

```text
h_0107
```

正式 evaluator 跑完后：

```text
objective = 0.00479
```

这说明这个 schedule_policy 改动产生了巨大提升。

reflection 会把这个记录成经验：

```text
stagnation_boost schedule helps utility_ratio family escape stagnation and reduce gap
```

下一代继续在这个结构上加 restart_policy、penalty_policy 微调。

这就是整个框架“像专家一样逐步改算法”的实际过程。

---

## A.20 整个 20 代最终发生了什么？

可以把最终有效路径理解为：

```text
初始 seed:
linear_usage + top_gap + periodic escape

第 1 代:
调 escape moves 到 3 -> h_0009, objective 0.15471

第 5-6 代:
加 penalty decay=0.1 -> h_0053, objective 0.13435

第 8 代:
换成 stagnation escape -> h_0071, objective 0.13435

第 10 代:
distance_update 改成 utility_ratio -> h_0097, objective 0.07947

第 11 代:
加入 stagnation_boost schedule -> h_0107, objective 0.00479

第 14 代:
加入 decay_penalty restart -> h_0140, objective 0.00456

第 16 代:
penalty edge_selection 改 tour_utility, top_k=15 -> h_0157, objective 0.00916, TSPLib 最好

第 19 代:
tour_utility top_k 从 15 降到 10 -> h_0183, objective 0.00437, training 最好
```

这个路径说明当前框架不是乱搜，而是在逐步组合有效机制。

---

## A.21 为什么说这个工作流像人类专家？

因为它有以下专家式动作：

| 人类专家行为 | 当前框架对应实现 |
|---|---|
| 先给几个初始算法 | expert bootstrap seeds |
| 每轮先想实验计划 | GenerationPlanner.plan |
| 每次只改一个因素 | validate_single_layer_change |
| 明确记录假设 | group_hypothesis / delta / expected_effect |
| 做正式实验 | evaluator.evaluate |
| 看平均值和坏例子 | training diagnostics |
| 总结哪些改动有效 | generation reflection |
| 把经验写进笔记 | expert_notebook / mechanism_archive |
| 下一轮基于经验继续 | planner reads notebook/archive |
| 不用失败结果糊弄过去 | no fallback / fail closed |

所以我们可以对导师这样解释：

```text
我们的框架不是一个普通进化算法，也不是单纯 LLM 代码生成。
它是一个带结构化实验计划、受控变量、正式评估、机制反思和长期记忆的自动算法设计系统。
```

---

## A.22 当前工作流中 LLM 到底负责什么？系统负责什么？

这是非常重要的区分。

### LLM 负责

```text
1. 根据当前 population 和历史经验提出下一代实验目标
2. 为每个 parent 设计一个 child_spec
3. 写明 group_hypothesis / delta / expected_effect
4. 每代结束后总结哪些机制有效/失败
```

### 系统负责

```text
1. 检查 child_spec 是否合法
2. 检查是否只改了一层
3. 根据 child_spec 确定性生成 Python 代码
4. contract check
5. 正式 evaluator 评估
6. 计算训练诊断
7. 更新 population
8. 保存所有日志和结果
```

这意味着：

```text
LLM 是算法设计者，系统是实验管理员和审稿人。
```

LLM 不能随意绕过规则。

---

## A.23 一个候选从出生到进入种群的数据流

以一个 offspring 为例，它的数据流是：

```text
offspring_plan
  -> child_spec
  -> validate child_spec
  -> build_arch_spec_code(child_spec)
  -> Candidate(id, parent, algorithm, code, metadata)
  -> contract_check
  -> evaluator.evaluate(code)
  -> objective
  -> training_diagnosis
  -> candidate_store
  -> mechanism_archive
  -> population_manager.update
  -> maybe survive into next generation
```

其中 `Candidate` 里会保存：

```text
id
parents
algorithm
code
objective
selection_score
mechanism_hypothesis
mechanism_profile
parent_delta
evidence
metadata.arch_spec
metadata.spec_diff
metadata.generation_plan
status
```

所以后续我们可以追踪：

```text
这个候选来自哪个 parent？
改了哪一层？
具体 delta 是什么？
训练表现如何？
尾部表现如何？
最后有没有活下来？
```

---

## A.24 本框架和 ReEVO 最大工作流区别

ReEVO 大致是：

```text
population of code snippets
  -> select parent pair
  -> short-term reflection
  -> LLM generates new code
  -> evaluate
  -> long-term reflection
  -> mutation
```

我们的框架是：

```text
population of auditable ARCH_SPEC algorithms
  -> select parent slots
  -> LLM plans controlled experiments
  -> child_spec must change exactly one layer
  -> deterministic code generation
  -> formal evaluation + diagnostics
  -> quality-diversity update
  -> generation-level reflection
```

关键区别：

```text
ReEVO 更像开放代码进化。
我们更像受控算法实验设计。
```

这就是为什么我们的实验更容易解释，也更接近导师关心的“人类专家设计算法过程”。

---

## A.25 如果导师问“你们到底创新在哪里？”

可以回答：

```text
第一，我们把 LLM 自动启发式设计从单函数代码搜索升级为 GLS 架构级搜索。

第二，我们没有让 LLM 任意写代码，而是让它输出可审计的 ARCH_SPEC。

第三，每个候选必须是受控实验，只能改变一个架构层，因此结果可以归因。

第四，我们引入训练诊断、机制 archive、expert notebook 和 generation reflection，让搜索过程带有长期专家记忆。

第五，我们坚持 fail-closed，没有 fallback，避免失败步骤污染实验。
```

这一套组合起来，就是当前框架的核心。
