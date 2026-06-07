# 聊天记录总结

## 项目目标

本项目是在原始 EoH 基础上构建一个并行的新框架 `eohagent`，目标是服务论文实验，而不是追求通用工程化。核心思路是保留 EoH 中已经证明有效的关键实现，例如算法个体生成方式、代码提取方式、评估接口、种群更新和 objective-level dedup，同时把创新点集中在 agent harness、skill、memory、compaction、generalization audit/probe 等机制上。

最终目标任务已经明确为原 EoH 的 TSP-GLS 问题，而不是早期误用的 `tsp_construct/select_next_node`。正确个体接口应为：

```python
def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    ...
    return updated_edge_distance
```

训练 fitness 使用 `examples/user_tsp_gls_OPRO/prob.py` 中的 `TSPGLS.evaluate(code_string)`，基于 `TrainingData/TSPAEL64.pkl` 返回 64 个训练实例平均 gap。最终测试使用 `examples/user_tsp_gls_OPRO/eval_tsp.py` 和 `TestingData/TSPLib/*.tsp`。

## 当前框架工作流

外层进化流程为：

```text
load config
-> bootstrap baseline seed
-> for each generation:
   -> rank selection 选择 parent
   -> agent harness 生成 offspring
   -> contract/audit 通过后正式 evaluator
   -> objective-level dedup
   -> population update
   -> 保存 population / memory / progress
```

每个 offspring 内部由 `eohagent/harness/agent_loop.py` 控制：

```text
AgentState 初始化
-> LLM 自主选择 skill
-> ToolRouter 执行 skill
-> AgentState 更新
-> CommitGate 判断下一步
-> commit / repair / reject / continue
```

这不是完全固定 workflow。LLM 可以选择 `mechanism_decomposition`、`controlled_variation`、`repair`、`notebook_update`、`probe_synthesis` 等 skill。但为了实验可靠性，`CommitGate` 会强制保证 candidate 在进入正式 evaluator 前通过 `contract_check` 和 `generalization_audit`；高风险候选会进入 repair 或 reject。

## Skill 含义

在当前框架里，skill 本质上是一段结构化 prompt/instruction，加上一些对应的执行逻辑。当前 skill 集合是预先写好的，不是运行时无限自动增长。

- `mechanism_decomposition`：分析 parent 算法机制，帮助 agent 理解当前个体为什么有效或可能失败。
- `controlled_variation`：基于 parent 生成新候选算法代码，是主要的 offspring 产生方式。
- `contract_check`：检查函数名、签名、返回值、语法和基本行为是否符合 TSP-GLS evaluator 要求。
- `repair`：当 contract 或 audit 失败时，尝试修复候选代码。
- `generalization_audit`：在正式评估前审查过拟合风险，目前主要是 AST/代码结构启发式。
- `probe_synthesis`：计划生成泛化探针，但当前实现还偏弱，主要返回 probe 描述，没有真正执行 probe。
- `context_compaction`：在 prompt 压力过大时压缩工作上下文，避免 Ollama 或 API 被动截断。
- `notebook_update`：把运行经验写入 expert notebook，但当前语义总结还不够强。

## 已完成的重要修改

1. 将 TSP 任务从错误的 `tsp_construct/select_next_node` 迁移到原 EoH 的 TSP-GLS `update_edge_distance`。
2. 学习原 EoH 的方式，减少对严格 JSON 的依赖，改为更鲁棒的自然语言和代码块提取。
3. 引入 baseline seed、problem card、candidate store、session log、expert notebook、compact context。
4. 保留 agent 自主选择 skill，但加入 evaluator 前的安全门控。
5. 增加并行评估，参考原 EoH 的并行方式，避免串行评估过慢。
6. 增加训练过程打印信息，包括 generation、offspring、candidate id、skill、eval start/done、objective duplicate 等。
7. 增加 prompt budget/context pressure 机制：发送 LLM 前估算 prompt 长度，超过阈值则强制 compaction，避免长 prompt 被模型服务截断。
8. 增加 DeepSeek OpenAI-compatible 配置，可以用 `deepseek-v4-flash` 跑同样的 20x20 实验。

## 关键实验结果

早期错误版本使用 `tsp_construct/select_next_node`，objective 在 `6.x` 路径长度尺度，不符合 EoH 的 TSP-GLS 任务，已经废弃。

正确迁移到 TSP-GLS 后，训练 objective 回到 EoH 风格的 gap 尺度，例如 `0.00575`、`0.00642`、`0.00822` 等。

DeepSeek v4 flash 20 代、种群 20 的最终训练种群中，训练最优为：

```text
h_0331: objective=0.00575
h_0332: objective=0.00642
h_0280: objective=0.00693
h_0231: objective=0.00740
h_0304: objective=0.00773
```

但 TSPLib 后验测试中，泛化最好的是：

```text
h_0270: train_obj=0.00822, tsplib_average_gap=0.2320148275862069
h_0231: train_obj=0.00740, tsplib_average_gap=0.23559827586206897
h_0280: train_obj=0.00693, tsplib_average_gap=0.23739655172413793
h_0386: train_obj=0.00972, tsplib_average_gap=0.2376651724137931
h_0354: train_obj=0.00865, tsplib_average_gap=0.23912241379310345
```

训练最优 `h_0331` 并不是测试最优。这说明当前框架已经能产生强训练个体，但还存在训练集 gap 和 TSPLib 泛化 gap 不完全一致的问题。这也正好暴露出下一步研究创新点：需要让 probe、audit、behavior signal 真正参与搜索。

## 当前主要问题

1. `probe_synthesis` 使用太少。统计中 `generalization_audit` 调用很多，但 `probe_synthesis` 只有少量调用，原因是 `audit_uncertain` 触发条件太窄。
2. probe 当前基本只是“描述”，还没有执行轻量测试，也没有产生可用于 selection 的 `probe_score`。
3. `generalization_audit` 主要是代码结构启发式，不能直接判断算法在 TSP-GLS 上是否更泛化。
4. `risk_weight` 已在配置中存在，但当前 selection 仍主要按 objective 排序，风险和 probe 没有真正影响个体保留。
5. expert notebook 已经存在，但内容更像日志摘要，还没有沉淀为高质量的“哪些机制有效、哪些机制应避免”的专家经验。
6. memory/compaction 已加入 prompt 预算控制，但还需要通过新一轮实验观察是否确实减少截断、重试和后期变慢。

## 对比原始 EoH 的认识

原始 EoH 有效的原因主要包括：

- 不强依赖严格 JSON，而是从自然语言回复中提取 algorithm 和 code。
- 强制产生新的算法个体，而不是只围绕 baseline 做保守小修。
- evaluator、训练数据、TSPLib 测试流程清晰稳定。
- population update 中有 objective-level dedup，避免大量等价个体占据种群。
- 进化主体简单直接，减少了无效中间环节。

因此 `eohagent` 的实现原则已经调整为：凡是 EoH 已经验证有效的底层机制，尽量迁移和保持一致；创新集中放在 agent 是否能利用 skill、memory、audit/probe 做出更好的搜索决策。

## 下一步研究方向

建议下一阶段聚焦“Probe-Guided Agentic EoH”，不要把 agent 退化成固定 workflow，而是给 agent 提供更强的证据工具：

1. 把 `probe_synthesis` 从返回 probe 描述升级为可执行轻量 probe。
2. 为每个 candidate 记录 `probe_score`、`behavior_signature` 和 `risk_score`。
3. 让 selection 不只看训练 objective，可以考虑：

```text
selection_score = train_objective + alpha * risk_score + beta * probe_penalty
```

4. 用 behavior-level 或 mechanism-level 多样性辅助 parent selection，避免所有个体收敛到相似机制。
5. 让 expert notebook 从 JSON 日志升级为语义经验库，例如记录“哪些 penalty 组合在 TSPLib 泛化更好”。
6. 做 ablation：EoH baseline、当前 agent、去掉 memory、去掉 audit、加入 executable probes 等。

## 已检索到的可借鉴 Skills

使用 `$find-skills` 检索后，比较相关的外部 skill 包括：

- `property-based-testing`：适合把泛化探针做成可执行 invariant/probe。
- `prompt-engineer`：适合改进 skill prompt、结构化输出和上下文管理。
- `jupyter-notebook`：适合整理实验结果、画图和生成汇报材料。
- `discover-research`：适合论文实验设计、统计分析和写作。
- `ai-research-skills`：包含机器学习论文写作、long-context、agent evaluation 等方向，可选择性借鉴。

这些 skill 不直接构成论文创新点，但可以帮助实现和验证创新点。当前最有论文价值的方向仍然是：让 agent 在 EoH 搜索中通过 memory、audit、probe 和 behavior signal 学会更好地控制泛化风险。

