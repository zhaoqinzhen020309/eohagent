# ARCH_SPEC 架构级 GLS 实验分析

实验目录：

```text
eohagent/results/tsp_gls_arch_spec_deepseek_10x20_v1
```

评估文件：

```text
eohagent/results/tsp_gls_arch_spec_deepseek_10x20_v1/population_generation_20_tsplib_eval.csv
eohagent/results/tsp_gls_arch_spec_deepseek_10x20_v1/population_generation_20_tsplib_eval.json
```

## 一、总体结论

这次结果是明确正向的。

本次最好个体：

```text
h_0157
training objective = 0.00916
TSPLib average gap = 0.21278068965517244
```

本地旧 DeepSeek 单函数框架最好结果：

```text
h_0270
training objective = 0.00822
TSPLib average gap = 0.2320148275862069
```

因此，这次 `ARCH_SPEC + 架构级 GLS 搜索` 在 TSPLib 上超过了旧 DeepSeek 单函数框架，约有 8.3% 相对提升。

这说明本次改动不是只让训练集分数变好，而是在最终测试集上也产生了更好的泛化结果。

## 二、训练过程是否有效

训练目标从初始种子开始持续下降：

```text
gen0  best=0.18540  avg=1.23446
gen1  best=0.15471  avg=0.28823
gen5  best=0.13733  avg=0.15297
gen10 best=0.07947  avg=0.12053
gen11 best=0.00479  avg=0.09473
gen20 best=0.00437  avg=0.00494
```

关键跳变发生在第 10 到第 11 代：

```text
gen10:
utility_ratio + penalty decay=0.1 + stagnation escape
best = 0.07947

gen11:
加入 schedule_policy = stagnation_boost
best = 0.00479
```

这说明本次搜索不是只在一个固定函数里调参数，而是通过多个 GLS 架构层的组合找到了强结构。

## 三、有效组件

### 1. ARCH_SPEC 有效

`ARCH_SPEC` 的作用是让候选算法不只是输出代码，还必须输出可审计的结构化算法规格。

这次 20 代中：

```text
planner 每代都成功返回 10 个计划
generation reflection 每代都成功
没有 generation_failed
没有 fallback
repair_used = 0
```

200 个计划子代中：

```text
evaluated = 178
failed = 22
repair_used = 0
```

失败的主要原因：

```text
16 x Duplicate ARCH_SPEC code generated
6 x Code evaluation failed
```

这说明严格合同确实在保护实验纯净性。重复架构没有被偷偷当成新实验进入评估，而是被直接拒绝。

### 2. 架构级 GLS 搜索有效

最终强结构不是原来的单函数 `update_edge_distance`，而是多层组合：

```text
distance_update_policy = utility_ratio
penalty_policy.edge_selection = tour_utility 或 top_gap
penalty_policy.decay = 0.1
escape_policy.moves = 3
restart_policy = decay_penalty
schedule_policy = stagnation_boost
```

其中有明确贡献的层包括：

```text
distance_update_policy: utility_ratio 最终胜出
penalty_policy: decay=0.1、tour_utility/top_gap 有效
escape_policy: moves=3 早期明显有效
restart_policy: decay_penalty 后期有效
schedule_policy: stagnation_boost 非常关键
```

这支持我们新的项目定位：

```text
从单函数启发式进化，升级为受约束的 GLS 架构级设计。
```

### 3. 质量多样性选择部分有效

最终训练分数最好的个体不是 TSPLib 最好的个体：

```text
h_0183 train=0.00437 TSPLib=0.22943
h_0157 train=0.00916 TSPLib=0.21278
```

`h_0157` 训练分数不是第一，但它被保留在最终种群里，并且 TSPLib 最好。

这说明 quality-diversity 选择不是完全无效。它至少保住了一个训练分数稍差、但测试泛化更好的结构。

## 四、无效或效果不足的组件

### 1. 缓存和压缩基本无效

20 代总计：

```text
prompt_tokens = 576944
completion_tokens = 118994
cache_hit_tokens = 28800
cache_hit_rate = 4.99%
compression_saved_chars = 17571
```

缓存命中率只有约 5%，压缩节省也很小。

这个组件目前不能算成功。如果 DeepSeek 成本是核心问题，这一块必须继续改。它不影响算法结果的正确性，但会影响实验成本。

### 2. 多样性保留还不够

最终 10 个个体虽然 ARCH_SPEC 都不同，但全部属于同一个机制家族：

```text
distance_update_policy = utility_ratio
mechanism_family = utility_ratio_penalty
```

也就是说，当前 quality-diversity 保留的是 `utility_ratio` 内部的小多样性，而不是机制家族级多样性。

这次没有导致结果失败，因为 `utility_ratio` 确实很强。但从科研框架角度看，它还没有完美模拟人类专家会保留多条机制路线的行为。

### 3. 控制变量实验有效，但仍不完美

有效点：

```text
每个候选都能检查是否只改了计划指定的一层。
```

不足点：

```text
后期计划仍然过度围绕当前最优父代微调。
重复 ARCH_SPEC 增多。
agent 对“当前方向是否已经饱和”的判断还不够主动。
```

人类专家在看到一个局部结构反复微调没有明显提升后，通常会主动转向另一组机制假设。当前 agent 还没有稳定做到这一点。

## 五、哪些组件不建议去掉

不建议去掉：

```text
ARCH_SPEC
workflow audit
严格合同检查
no fallback 规则
generation planning
generation reflection
restart_policy
schedule_policy
quality-diversity survival selection
```

原因是这些组件共同构成了这次成功的基础。

特别是 `schedule_policy` 和 `restart_policy`，虽然它们在早期看起来不一定明显，但后期强结构中它们确实参与了最终提升。

## 六、哪些组件应该改进

### 1. prompt compression 需要重做

当前压缩收益太低，不建议继续指望现在这套压缩逻辑显著降低 DeepSeek 成本。

更合理的方向是重新设计 prompt 结构，让稳定前缀更长、动态信息更短，并减少每代重复发送的大段上下文。

### 2. duplicate ARCH_SPEC 抑制需要前置

现在 duplicate 检查是在候选生成后拒绝。

这能保护实验纯净性，但会浪费 LLM 调用和评估机会。

后续可以让 planner 输入显式包含 forbidden specs 或 forbidden niche，让它在计划阶段就避免重复架构。

### 3. quality-diversity 需要机制家族级增强

当前最终种群仍然塌缩到 `utility_ratio` 家族。

后续应该让质量多样性选择至少保留：

```text
机制家族级 champion
架构 niche champion
训练性能 champion
```

这样更接近人类专家同时维护多条候选路线的过程。

### 4. 训练-测试错位仍需处理

最终训练最优不是 TSPLib 最优：

```text
训练最优: h_0183 / h_0192 / h_0193 / h_0198
TSPLib 最优: h_0157
```

这说明训练目标仍然不能完全代表泛化性能。

这不是本次框架失败，而是说明下一版需要继续提升训练内诊断、失败模式建模和泛化风险识别。

## 七、是否完美模拟了人类专家设计算法

还没有完美实现，但比前几版明显更接近。

目前已经做到：

```text
每代先生成实验计划
每个候选有明确 parent + delta
候选只能改指定架构层
失败候选严格失败
代末做 reflection
最终能从单函数优化升级到 GLS 架构组合设计
```

这些已经符合人类专家做启发式算法设计的基本流程。

但还缺三点：

```text
1. 专家会主动判断“这个方向是否饱和”，当前 agent 还不够主动。
2. 专家会保留机制家族级备选路线，当前最终仍塌缩到 utility_ratio。
3. 专家会更重视泛化风险，当前训练最优和 TSPLib 最优仍不一致。
```

因此，这版框架不是终局版本，但方向是对的。

## 八、最终判断

当前最有效的核心创新是：

```text
受约束 ARCH_SPEC 架构搜索
+ 严格单层变更审计
+ no fallback
+ 质量多样性保留
```

当前最需要继续改的是：

```text
机制家族级多样性
重复规格抑制
训练-测试错位
缓存成本
```

这次实验最重要的结果是：

```text
ARCH_SPEC 架构级设计路线已经超过本地旧 DeepSeek 单函数 EoH 结果。
```

这说明“从固定函数设计升级到受约束的 GLS 架构级设计”是值得继续推进的方向。
