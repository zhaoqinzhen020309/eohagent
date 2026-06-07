# 泛化导向专家式 Harness 设计

日期：2026-06-05

## 目标

这份设计用于修改当前 EoH agent 框架，让搜索过程更像一个有实验纪律的人类算法专家：先诊断失败，再提出受控修改，提前拒绝无效实验，避免重复走已知无效方向，并且在选择候选时使用训练实例内部的稳健性信号。

核心目标是提升泛化能力，同时保持实验协议干净：

- 训练实例只用于进化过程。
- 最终测试实例只用于最终评估。
- 不引入额外 probe 集。
- 不允许 fallback 机制。

## 外部依据

本设计受到以下工作的启发：

- AutoEoH：使用任务检查、评价检查、结果检查等流程组织自动 EoH，https://github.com/FeiLiu36/AutoEoH
- A2DEPT：强调结构化进化、层级操作和程序维护循环，https://arxiv.org/abs/2604.24043
- EoH-S：使用实例级表现证据讨论单目标启发式进化的泛化问题，https://arxiv.org/abs/2508.03082
- Quality-Diversity / MAP-Elites：不只保留单个贪心最优，而是保留高质量且多样的解，https://arxiv.org/abs/1504.04909
- 自适应算子选择相关文献可以作为背景，但本设计明确不实现单独的算子概率自适应模块。

## 已确认范围

只实现中等强度的搜索空间自适应方案，也就是前面讨论的 B 方案：

- 允许框架在已有 typed 搜索空间里禁用已知无效修改模式。
- 不实现纯粹的算子概率调度。
- 不允许 planner 发明新的 typed component。
- 不引入运行时算法组合或自动启发式选择器。
- 一次运行可以保存最终种群，但最终测试仍然可以只选择一个候选启发式算法。

这意味着：搜索空间不会动态扩展，只会被动态约束。

## 主流程

每一代按以下流程执行：

1. 构建精简的专家诊断包。
2. 要求 planner 生成短格式、受控的实验计划。
3. 在正式评估前进行确定性实验审查。
4. 无效 offspring 直接拒绝，不 repair，不 fallback。
5. 有效 offspring 只在训练实例上评估。
6. 计算训练稳健性诊断。
7. 根据确定性 checker 证据更新禁用模式。
8. 使用统一鲁棒排序选择幸存个体和父代。
9. 执行短格式结构化 LLM 反思。

## 专家式 Harness

Harness 的目的不是让 LLM 更自由，而是让 LLM 像专家一样设计实验，而不是随机改配置。

### Planner 输入

Planner 不应该接收完整日志，而应该接收精简的专家诊断包。诊断包包含：

- 当前最优候选摘要。
- 最差训练实例摘要。
- 当前机制家族拥挤情况。
- 最近被禁用的修改模式。
- 最近被支持和被反驳的短假设。
- 最近无效或重复的搜索方向。
- 当前 typed 搜索空间约束。

这些内容必须严格限制长度，避免 token 成本和上下文干扰。

### Planner 输出

每个 offspring plan 必须包含一个很短的实验假设：

```json
{
  "target_failure": "tail_heavy_training_failures",
  "hypothesis": "reduce over-penalizing hard instances",
  "controlled_component": "penalty_policy",
  "controlled_change": "edge_selection",
  "expected_metric": "tail_gap",
  "child_spec": {}
}
```

这个假设只用于实验归因和审查，不要求长推理，不允许自由长文，也不允许一次计划里包含多个不受控修改。

## 实验审查

Planner 输出后，正式评估前必须进行确定性实验审查。

这个审查必须是一个明确的 pre-eval gate，而不是散落在多个步骤里的隐含检查。它的作用是决定一个 offspring 是否有资格进入正式训练评估。

审查输入包括：

- parent id。
- parent spec。
- planner 输出的短实验假设。
- planner 输出的 child spec。
- 当前 run 已禁用的修改模式。
- problem adapter 提供的 schema、diff 规则和语义 no-op 规则。

审查输出必须结构化记录：

```json
{
  "passed": true,
  "rejection_reason": null,
  "changed_layers": ["penalty_policy"],
  "changed_fields": ["penalty_policy.edge_selection"],
  "semantic_change": "confirmed",
  "disabled_pattern_hit": null
}
```

失败时也必须记录同样结构，只是 `passed=false`，并写清楚拒绝原因。

审查内容包括：

- child spec 是否符合 typed schema。
- planner 声明的 controlled_component / controlled_change 是否和实际 spec 差异一致。
- child 是否只修改了允许修改的字段。
- child 是否触碰了当前 run 的禁用模式。
- parent 到 child 的修改是否有实际语义变化。
- 这个实验是否足够受控，可以归因结果。
- 如果使用确定性代码包装，生成出来的候选代码是否严格对应审查通过的 child spec。

如果审查失败：

- 拒绝该 offspring。
- 记录明确失败原因。
- 不 repair。
- 不生成 fallback child。
- 不静默修改 planner 的计划。

对 ARCH_SPEC 这类 typed spec 工作流，contract check 失败也不能进入 repair skill。失败就是拒绝 offspring 并记录原因。否则 repair 可能改掉原本的实验假设，污染“这个受控修改到底有没有用”的归因。

如果某个问题 adapter 暂时没有提供语义 no-op 规则，框架不能自己猜测语义等价，也不能因为 fitness 不提升而禁用方向。此时只能执行 schema 和 diff 审查，并把语义审查状态记录为 unknown。

## 禁用模式

禁用模式指当前 run 中某个具体无效修改模式。

规则：

- 第一次确认语义无效，就立刻禁用该模式。
- 禁用范围是当前 run 的所有 parent。
- 禁用具体无效模式，不禁用整个 layer。
- 不能因为 fitness 没提升就禁用某个模式。
- 只有确定性的语义证据才能触发禁用。
- 如果 pre-eval 审查确认是语义 no-op，拒绝这个 offspring，并立即把对应具体模式加入当前 run 的禁用列表。

示例：

```text
无效模式：
  moves == max_moves 时，只修改 escape_policy.trigger

仍然允许：
  修改 moves 或 max_moves
  同时修改 trigger 和能让行为真正变化的字段
```

## 统一鲁棒排序

框架不使用类似“60% 主力、20% 尾部修复、20% 多样性”的手调槽位规则。

候选选择统一使用一个排序机制。这个排序同时用于：

- 谁留在下一代。
- 谁更容易作为父代。

排序只使用训练集内部诊断，包括：

- 平均 gap。
- 最差一批训练实例的 gap。
- 单个最差训练实例的 gap。
- 不同训练实例之间的波动。
- 机制家族拥挤软惩罚。

测试集不能影响这个排序。

第一版实现应尽量避免手调权重。优先使用 Pareto 风格或元组排序方式同时考虑 mean、tail、max、std。机制家族拥挤只作为软惩罚或 tie-breaker，不做硬比例限制。

## 机制家族趋同

后期个体趋同通过机制家族拥挤软惩罚来缓解。

规则：

- problem adapter 提供候选所属机制家族标签。
- 过度拥挤的家族在排序中会吃亏。
- 如果某个拥挤家族的候选确实很强，仍然可以保留下来。
- 不设置硬性的 family 数量上限。

这样可以抑制坍缩，但不会误杀真正强的机制家族。

## 训练诊断

泛化压力只来自训练实例内部诊断。

Evaluator evidence 应包含每个训练实例的 gap。框架基于它计算：

- mean_gap。
- tail_gap。
- max_gap。
- std_gap。
- 相比 parent 改善的训练实例。
- 相比 parent 恶化的训练实例。
- failure signature。

这些诊断用于鲁棒排序和反思，不使用最终测试实例。

## 结构化反思

保留 LLM reflection，但必须短且结构化。

允许格式：

```json
{
  "should_try": ["short rule"],
  "should_avoid": ["short rule"],
  "unresolved_failures": ["short name"],
  "supported_hypotheses": ["candidate: short result"],
  "refuted_hypotheses": ["candidate: short result"]
}
```

限制：

- 每个列表最多 3 条。
- 每条内容都必须很短。
- Reflection 只是建议。
- 硬性 disabled patterns 只能来自确定性 checker 证据。
- 鲁棒排序只能来自确定性指标。

## 通用框架边界

核心框架必须保持问题无关。

核心框架负责：

- 构建专家诊断包。
- 执行实验审查协议。
- 维护禁用模式状态。
- 执行统一鲁棒排序。
- 约束结构化 reflection 格式。
- 保证 fail-closed 行为。

Problem adapter 负责：

- typed 搜索空间。
- 候选机制家族标签。
- 语义 no-op 规则。
- 训练实例评价证据。
- 问题特定 typed schema 验证。

这样后续 TSP、CVRP、FSSP 都可以接入，而不会把 TSP 当前五层 ARCH_SPEC 写死进核心框架。

## 非目标

本设计不实现：

- probe 集评价。
- 测试集反馈进搜索。
- 运行时 heuristic portfolio 选择。
- planner 创建新的 typed component。
- 独立的算子概率自适应模块。
- fallback child 生成。
- planner/checker/evaluator 失败后的自动 repair。
- 长自然语言 reflection notebook。

## 预期代码位置

可能涉及：

- `eohagent/core/engine.py`：实验审查、选择排序、禁用模式接入。
- `eohagent/core/population.py`：鲁棒幸存选择，以及和父代选择一致的排序。
- `eohagent/harness/generation_planner.py`：专家诊断包、短假设格式、禁用模式注入。
- `eohagent/memory/memory_manager.py`：精简结构化反馈和机制拥挤诊断。
- `eohagent/core/training_diagnostics.py`：确保实例级鲁棒诊断完整且精简。
- 具体 problem adapter，例如 `eohagent/problems/tsp_gls_arch_spec`：typed schema、family 标签、语义 no-op 规则。

## 验收标准

实现完成必须满足：

- 语义无效模式在评估前被拒绝。
- 第一次确认语义 no-op 后，具体模式在本次 run 后续全部禁用。
- 禁用模式传入 planner，并在 planner 输出后强制执行。
- Candidate plan 包含短实验假设。
- Reflection 输出短且结构化。
- 幸存选择和父代选择使用同一个训练集鲁棒排序。
- 最终测试数据永远不参与进化。
- 不引入 fallback 或自动 repair 路径。
