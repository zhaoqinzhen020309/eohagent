# 给导师汇报用：当前框架高层工作流说明

> 这份文档只讲“大框架”和“信息如何流动”，不讲代码实现细节。  
> 适合直接转成 PPT。  
> 重点回答：LLM 做什么？Agent 做什么？信息怎么存？上下文怎么压缩？当前框架还有什么问题？

---

## 1. 我们现在的框架到底是什么？

当前框架可以概括为：

```text
一个由 LLM 驱动的专家式算法设计 Agent，
通过结构化算法规格 ARCH_SPEC 来设计 TSP-GLS 启发式算法，
并通过评估、记忆、反思、上下文压缩形成多代迭代优化闭环。
```

它不是简单的：

```text
LLM 写代码 -> 跑一下 -> 选最好的
```

而是更接近：

```text
LLM 扮演算法专家
Agent 扮演实验管理员
Evaluator 扮演实验平台
Memory 扮演实验笔记本
Context Manager 扮演信息压缩器
```

整体目标是模仿人类专家设计算法的过程：

```text
观察结果 -> 提出假设 -> 设计受控实验 -> 执行评估 -> 总结经验 -> 下一轮改进
```

---

## 2. 高层工作流总览

可以用下面这张流程图理解：

```text
        ┌──────────────────────┐
        │  Expert Seeds         │
        │  初始专家算法规格      │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Formal Evaluator     │
        │  评估初始种子          │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Population           │
        │  当前候选算法种群       │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Context Builder      │
        │  构造给 LLM 的上下文    │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  LLM Planner          │
        │  设计一代受控实验       │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Agent / Guardrails   │
        │  检查计划是否合法       │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  ARCH_SPEC → Code     │
        │  结构化规格转代码       │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Formal Evaluator     │
        │  训练集正式评估         │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Diagnostics          │
        │  训练诊断与失败模式      │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Memory Update        │
        │  写入记忆和归档         │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  LLM Reflection       │
        │  总结本代经验           │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Context Compression  │
        │  压缩历史信息           │
        └──────────┬───────────┘
                   │
                   └──────> 下一代
```

一句话：

```text
LLM 负责提出实验假设和总结经验；
Agent 负责把 LLM 的想法约束成合法实验；
Evaluator 负责给出真实反馈；
Memory 和 Context Manager 负责让系统长期记住关键经验但不爆上下文。
```

---

## 3. LLM 在框架中扮演什么角色？

当前 LLM 主要扮演两个角色。

---

### 3.1 角色一：实验规划者 Planner

每一代开始时，LLM 不直接写代码，而是先输出一个“实验计划”。

它要回答：

```text
这一代应该测试什么机制？
哪些方向应该避免重复？
每个父代应该改哪一个组件？
这个改动预期会改善什么？
```

LLM 输出的是结构化 JSON，例如：

```text
generation_goal
avoid_repeating
offspring_plans
```

每个 offspring plan 包含：

```text
parent_id
controlled_component
delta
expected_effect
child_spec
```

其中 `child_spec` 是完整的 GLS 算法架构规格。

这意味着：

```text
LLM 的任务是设计实验，而不是随便写代码。
```

---

### 3.2 角色二：实验反思者 Reflector

每一代评估结束后，LLM 会总结本代实验结果。

它要回答：

```text
哪些机制被验证有效？
哪些假设失败了？
哪些失败模式反复出现？
下一代应该继续探索什么？
哪些方向应该避免？
```

输出会写入 expert notebook，例如：

```text
validated_principles
failed_hypotheses
supported_deltas
refuted_deltas
open_questions
next_generation_priorities
```

这相当于人类专家做完一轮实验后写实验笔记。

---

## 4. Agent 在框架中扮演什么角色？

如果说 LLM 是“想法来源”，Agent 就是“实验管理员 + 审稿人”。

Agent 不盲目相信 LLM，而是对 LLM 输出做严格约束。

它主要负责：

```text
1. 检查 LLM 输出是不是合法 JSON
2. 检查每个 offspring 是否对应指定 parent
3. 检查 child_spec 是否完整
4. 检查是否只改变一个架构层
5. 检查是否重复已有算法
6. 把合法 ARCH_SPEC 转换成代码
7. 调用 evaluator 评估
8. 记录日志、诊断和失败原因
9. 更新 population
10. 触发 reflection 和 compaction
```

所以我们可以这样解释：

```text
LLM 是算法设计专家；
Agent 是实验流程控制器，保证 LLM 的设计变成可评估、可审计、可归因的实验。
```

---

## 5. 信息在系统中怎么流动？

系统中有几类信息。

---

### 5.1 当前种群信息 Population

每一代保留 10 个当前候选算法。

每个候选记录：

```text
算法结构
训练分数
父代是谁
改动了哪一层
机制假设
训练诊断
失败模式
```

这些信息告诉 LLM：

```text
当前有哪些算法？
哪个最好？
哪些机制已经试过？
哪些失败了？
```

---

### 5.2 评估反馈 Evaluation Feedback

每个候选评估后，不只记录平均 gap，还记录：

```text
mean_gap
std_gap
tail_gap
max_gap
worst_instances
failure_signature
mean_delta_vs_parent
tail_delta_vs_parent
```

这比单一分数更接近专家看实验结果的方式。

例如专家不会只问：

```text
平均性能变好了吗？
```

还会问：

```text
是不是尾部实例变坏了？
是不是某个实例崩了？
这个改动相对父代到底改善了什么？
```

---

### 5.3 Expert Notebook

这是 LLM reflection 产生的长期经验。

里面记录：

```text
已验证原则
失败假设
反模式
有用 delta
开放问题
下一步建议
```

可以理解为：

```text
专家实验笔记本。
```

---

### 5.4 Mechanism Archive

这是按机制归档的长期记忆。

它关注：

```text
哪个机制家族表现好？
哪个机制被反复验证？
哪个机制被反复否定？
某个机制家族的 best candidate 是谁？
```

例如：

```text
utility_ratio_penalty 家族目前最好
linear_usage_penalty 早期有用但后期被超越
sqrt/log usage 容易出现 extreme failure
```

---

### 5.5 Compact Context

这是给下一代 LLM 使用的压缩上下文。

它不是完整历史，而是提炼后的重点：

```text
current_best
recent_successes
recent_failures
design_lessons
mechanism_cards
delta_cards
anti_patterns
next_focus
working_memory
```

可以理解为：

```text
把完整实验历史压缩成下一代 LLM 需要看的摘要。
```

---

## 6. 我们的上下文压缩用的是什么框架？

这个问题要讲清楚：

```text
我们没有使用外部的 LangChain Memory、MemGPT、RAGFlow 或向量数据库框架。
```

当前上下文管理是项目内自研的，主要由四层组成：

```text
1. 分层记忆 Hierarchical Memory
2. 确定性裁剪 Deterministic Bounded Context
3. 可选 LLM Compaction
4. Prompt Compression / Token Cost Audit
```

---

### 6.1 第一层：分层记忆 Hierarchical Memory

系统不是把所有历史都塞给 LLM，而是分成几类记忆：

| 记忆类型 | 作用 |
|---|---|
| session_log | 原始事件日志，记录每一步发生了什么 |
| candidate_store | 保存所有候选完整信息 |
| expert_notebook | 保存 LLM reflection 得到的专家经验 |
| mechanism_archive | 按机制家族归档成功/失败假设 |
| compact_context | 给下一次 prompt 用的压缩摘要 |

这样做的好处是：

```text
完整历史还在硬盘里，但每次 prompt 只读取最重要的摘要。
```

---

### 6.2 第二层：确定性裁剪 Deterministic Bounded Context

每次构造 prompt 前，系统估算 prompt 长度：

```text
estimated_tokens ≈ prompt_chars / chars_per_token
```

当前配置：

```yaml
context_window: 65536
compact_threshold: 0.70
chars_per_token: 3.0
```

也就是说，如果 prompt 预计超过上下文窗口的 70%，系统会触发上下文压缩。

压缩时，系统会确定性保留：

```text
当前最好个体
少量 population summary
最近成功/失败事件
少量 expert notebook 条目
少量 mechanism archive 条目
next search hint
```

并裁掉：

```text
大量旧候选完整代码
太久以前的事件
冗长的历史 notebook
完整 archive
```

这是一种规则驱动的 bounded context，不依赖 LLM。

---

### 6.3 第三层：可选 LLM Compaction

代码里支持 LLM compaction：

```text
让 LLM 把历史压缩成 resume_summary、keep_rules、avoid_rules、next_search_hint 等。
```

但本次实验配置里：

```text
use_llm_compaction = false
```

所以本次主要使用的是 deterministic bounded context，而不是每次都让 LLM 总结上下文。

这点要讲清楚，避免导师以为我们用了一个复杂外部 memory framework。

---

### 6.4 第四层：Prompt Compression

除此之外，还有一层 prompt compression。

它不是“语义记忆压缩”，而是对 prompt 中较长的 JSON 内容做字符级/leaf-value 压缩，目标是减少发送给 LLM 的字符数。

本次实验中它确实开启了：

```yaml
prompt_compression:
  enabled: true
  threshold_chars: 8000
  rate: 0.55
```

但是效果一般。

本次统计：

```text
prompt_tokens = 576944
completion_tokens = 118994
cache_hit_tokens = 28800
cache_hit_rate ≈ 4.99%
compression_saved_chars = 17571
compression_saved_ratio ≈ 0.95%
```

说明当前 prompt compression 对成本帮助很小。

---

## 7. 当前上下文管理的问题是什么？

### 7.1 问题一：压缩节省太少

本次 compression saved ratio 只有约 0.95%。

说明当前压缩主要是在已有 prompt 上做局部缩短，没有从结构上减少信息重复。

导师可以理解为：

```text
我们现在的压缩更像“剪短句子”，而不是“重构信息流”。
```

后续需要把 prompt 结构分成：

```text
稳定长期规则
当前代动态信息
可引用的候选 ID
按需展开的详细信息
```

而不是每次把大量动态 JSON 都发给 LLM。

---

### 7.2 问题二：缓存命中率低

DeepSeek cache hit rate 只有约 5%。

原因是每代 prompt 中动态内容太多：

```text
population 变化
training diagnostics 变化
notebook 变化
archive 变化
recent events 变化
```

这些都会破坏前缀缓存。

也就是说：

```text
不是模型缓存机制没用，而是我们的 prompt 结构不够 cache-friendly。
```

---

### 7.3 问题三：compact context 还不够“专家化”

当前 compact context 主要保留：

```text
recent_successes
recent_failures
design_lessons
mechanism_cards
anti_patterns
next_focus
```

这已经有用，但还不够像专家的“研究计划”。

专家可能会组织成：

```text
当前主假设
备选假设
已饱和方向
下一代必须验证的问题
不能再试的重复方向
泛化风险提醒
```

当前 compact context 对这些信息的结构化还不够强。

---

### 7.4 问题四：压缩没有解决泛化问题

上下文压缩主要解决 prompt 长度和成本，不直接解决算法泛化。

当前仍然出现：

```text
训练最优 h_0183: TSPLib = 0.22943
测试最优 h_0157: TSPLib = 0.21278
```

说明框架还需要更强的泛化诊断，而不是只靠压缩历史信息。

---

## 8. 当前框架哪里做得好？

### 8.1 LLM 不是直接写代码，而是做实验设计

这点是核心优势。

它让搜索过程更可控、更可解释。

```text
LLM 输出机制假设和 ARCH_SPEC；
系统负责生成代码和验证。
```

---

### 8.2 信息闭环已经形成

当前已经有完整闭环：

```text
实验计划 -> 评估 -> 诊断 -> 反思 -> 记忆 -> 下一代计划
```

这比普通进化算法多了“语义反思”和“机制记忆”。

---

### 8.3 结果能证明框架有效

训练上：

```text
best objective: 0.18540 -> 0.00437
```

测试上：

```text
ARCH_SPEC TSPLib best = 0.21278
ReEVO adapted TSPLib best = 0.34209
```

说明不是空跑，而是确实找到了更好的 GLS 结构。

---

### 8.4 可解释性强

最终算法不是一个黑盒代码片段，而是可以解释为：

```text
utility_ratio distance update
+ penalty decay
+ escape moves=3
+ decay_penalty restart
+ stagnation_boost schedule
```

这很适合给导师讲。

---

## 9. 当前框架哪里还有问题？

### 9.1 机制家族多样性不足

最终种群几乎都集中到：

```text
utility_ratio_penalty
```

这说明 quality-diversity 还不够强。它保留了同一机制内部的小差异，但没有保留足够多的不同大方向。

后续需要：

```text
机制家族级 champion
架构 niche champion
泛化风险 champion
```

---

### 9.2 Planner 后期容易局部微调

后期 planner 反复围绕当前最优结构做细节调整，导致重复 ARCH_SPEC 增加。

本次：

```text
Duplicate ARCH_SPEC = 16
```

说明 planner 对“这个方向是否已经饱和”的判断还不够。

---

### 9.3 上下文压缩成本收益低

前面已经说过：

```text
cache hit ≈ 5%
compression saved ≈ 0.95%
```

如果未来跑更大实验，这会带来成本问题。

---

### 9.4 Reflection 有用但有噪声

LLM reflection 能总结有效经验，但有时会出现上下文依赖没有讲清楚的问题。

例如某个机制在一个父代上有效，在另一个父代上可能无效。当前 notebook 有时会把它总结得太绝对。

后续应该让 reflection 记录：

```text
机制在哪个上下文下有效
和哪些组件组合时有效
在哪些失败模式下无效
```

---

### 9.5 泛化诊断还不够

训练最优不等于测试最优，说明当前 selection 仍然偏向训练集。

后续需要引入更好的泛化风险代理，例如：

```text
训练实例分组
tail risk
失败模式多样性
小规模 held-out probe
```

但要注意不能污染最终 TSPLib 测试。

---

## 10. 给导师汇报时可以怎么讲？

可以这样讲当前框架：

```text
我们现在不是让 LLM 直接生成启发式代码，
而是构建了一个专家式算法设计 Agent。

这个 Agent 每一代先让 LLM 设计一组受控实验，
每个实验只改变 GLS 架构中的一个层，
然后系统把结构化规格转成代码并正式评估。

评估结果不仅包含平均性能，还包含尾部风险和失败模式。
这些信息被写入专家笔记和机制归档，
下一代 LLM 再基于这些长期记忆继续规划。

为了防止上下文越来越长，我们实现了自研的分层记忆和 bounded context 压缩。
目前这套压缩能保证实验运行，但成本收益还不高，cache 命中率也低，
这是下一步需要改进的工程和方法问题。
```

---

## 11. 一页 PPT 版总结

如果只放一页，可以写：

```text
Framework = LLM Planner + Agent Guardrails + Formal Evaluator + Long-term Memory + Context Compression

1. LLM Planner:
   设计每一代受控 ARCH_SPEC 实验

2. Agent Guardrails:
   检查每个实验只改变一个 GLS 架构层，防止无效/重复/污染候选

3. Formal Evaluator:
   在 64 个训练实例上评估 mean gap，并记录 tail gap / failure signature

4. Long-term Memory:
   expert notebook + mechanism archive 保存有效机制、失败假设和下一步建议

5. Context Compression:
   自研分层记忆 + deterministic bounded context，控制 prompt 长度

6. Current Problems:
   机制家族多样性不足；planner 后期局部微调；压缩和缓存收益低；泛化诊断仍需增强
```

---

## 12. 最核心的贡献表述

可以对导师这样概括：

```text
我们的贡献不是简单换了一个搜索算法，
而是把 LLM 自动算法设计组织成了一个可审计的专家式实验循环。

LLM 负责提出机制假设，
Agent 负责保证实验合法和可归因，
Evaluator 负责提供真实反馈，
Memory 和 Context Manager 负责让系统跨代积累经验。

当前实验结果说明，这种架构级、受控实验式的自动算法设计，
比传统 ReEVO-style 单函数代码搜索更适合当前 TSP-GLS 任务。
```
