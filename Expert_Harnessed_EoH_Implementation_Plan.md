# Expert-Harnessed EoH：基于 EoH 的专家式 Agent Harness 重写方案

> 目标读者：Codex / 代码实现智能体  
> 目标仓库：https://github.com/FeiLiu36/EoH  
> 核心要求：不要在原 `eoh/` 工程里继续打补丁式新增 method。请在仓库根目录下创建一个与 `eoh/` 并行的新目录 `expert_eoh/`，重写核心 agentic evolution 逻辑；只复用或桥接 EoH 中成熟的任务接口、evaluator、种群管理思想、seed 初始化思想和结果保存思想。

---

## 0. 总体目标

我们要实现一个新的 LLM+EC 自动启发式算法设计框架，暂名：

```text
Expert-Harnessed EoH
或
BSA-EoH: Baseline-Seeded Agentic Evolution of Heuristics
```

这个框架基于 EoH 的大思想：

```text
heuristic = thought + code + fitness
```

但不继续沿用 EoH 的固定 prompt operators：

```text
E1 / E2 / M1 / M2 / M3
```

新的核心思想是：

```text
用一个 baseline heuristic 作为冷启动 seed，
把 baseline 转换成 expert notebook / problem card / initial candidate memory，
然后用一个带 skills、memory、context compaction、commit gate 的 agent harness，
模仿人类专家逐步分析、修改、测试、修复、总结启发式算法的过程。
```

简单说：

```text
EoH 原始模式：
    fixed prompt operator → generate thought/code → evaluate → select

新框架模式：
    baseline seed → expert bootstrap → agent loop with skills → commit/reject → evaluate → select → memory/compaction
```

---

## 1. 设计原则

### 1.1 不改任务接口

不要把 EoH 的任务接口改成所谓“通用接口”。

EoH 本来就是：用户给定某个任务的 heuristic 函数接口，框架在这个接口里进化代码。例如：

```text
BPP: heuristic(item, bins) -> scores
TSP / GLS: update_edge_distance(...) -> updated_edge_distance
FSSP / PFSP: update_time_matrix(...) 或其他任务给定接口
```

本项目继续保留这个设计。创新点不是接口，而是 **offspring generation 的 agent harness**。

### 1.2 不在原 EoH 里新增 method

不要做：

```text
eoh/src/eoh/methods/eoh/ 里面新增 agent operator
```

而是创建新目录：

```text
expert_eoh/
```

原 EoH 只作为：

```text
1. baseline 对照
2. evaluator / problem interface 来源
3. 种群管理思想参考
4. seed 初始化思想参考
```

### 1.3 baseline seed 是冷启动核心

不要让 LLM 从空白开始生成第一代。第一代必须至少包含一个简单、稳定、可解释的 baseline heuristic。

例如：

```text
BPP: First-Fit / Best-Fit
TSP: Nearest Neighbor / Nearest Insertion
PFSP: NEH-like rule
MKP: profit/weight greedy
```

baseline seed 不是作弊。它模拟人类专家的真实起点：专家通常从一个简单可用的启发式开始，再逐步修改。

### 1.4 agent 不是固定 workflow

不要实现成：

```text
mechanism_decomposition → controlled_variation → contract_check → repair → commit
```

这种固定线性流程。

要实现成真正的 bounded agent loop：

```text
while budget not exhausted:
    observe current state
    choose next skill
    execute skill/tool
    update memory/state
    maybe compact context
    commit / repair / reject / continue
```

每个 offspring 的生成路径可以不同。

### 1.5 memory 和 context compaction 是核心组件

不能把所有历史都塞进 prompt。必须有：

```text
session_log.jsonl
candidate_store.json
expert_notebook.json
compact_context.json
```

其中：

```text
memory = 长期状态
compact_context = 每次喂给 LLM 的压缩视图
```

---

## 2. 推荐工程结构

在仓库根目录新建：

```text
expert_eoh/
├── main.py
├── config/
│   ├── default.yaml
│   ├── bp_online.yaml
│   ├── tsp_construct.yaml
│   └── user_task_template.yaml
├── core/
│   ├── engine.py
│   ├── population.py
│   ├── selection.py
│   ├── evaluator_bridge.py
│   ├── candidate.py
│   └── result_store.py
├── bootstrap/
│   ├── baseline_loader.py
│   ├── expert_bootstrap.py
│   └── problem_card.py
├── harness/
│   ├── agent_loop.py
│   ├── agent_state.py
│   ├── skill_registry.py
│   ├── tool_router.py
│   └── commit_gate.py
├── skills/
│   ├── mechanism_decomposition/
│   │   └── SKILL.md
│   ├── controlled_variation/
│   │   └── SKILL.md
│   ├── contract_check/
│   │   └── SKILL.md
│   ├── repair/
│   │   └── SKILL.md
│   ├── generalization_audit/
│   │   └── SKILL.md
│   ├── probe_synthesis/
│   │   └── SKILL.md
│   ├── context_compaction/
│   │   └── SKILL.md
│   └── notebook_update/
│       └── SKILL.md
├── memory/
│   ├── session_log.py
│   ├── candidate_store.py
│   ├── expert_notebook.py
│   ├── compact_context.py
│   ├── compactor.py
│   └── memory_manager.py
├── llm/
│   ├── client.py
│   ├── json_call.py
│   └── schemas.py
├── seeds/
│   ├── bp_online_baseline.json
│   ├── tsp_baseline.json
│   └── mkp_baseline.json
└── results/
```

---

## 3. 与原 EoH 的关系

### 3.1 可以复用或桥接的部分

请优先参考 / 复用 / 桥接这些思想：

```text
1. EoH 的 problem/evaluator 接口
2. EoH 的 seed 初始化思想
3. EoH 的 population 管理思想
4. EoH 的 parent selection 思想
5. EoH 的结果保存思想
```

原 EoH README 中已经说明：使用 examples 时，要把 heuristic 复制到 `heuristic.py`，函数名、输入、输出必须和 evaluation block 对齐。新框架应保持这种评价协议。

### 3.2 应该淘汰 / 不再作为主流程的部分

不要继续使用：

```text
E1 / E2 / M1 / M2 / M3 fixed prompt operators
InterfaceEC._get_alg() 那种 operator 分发逻辑
原始 prompt strategy 体系
```

这些只可以作为实验 baseline 或 ablation，不进入新框架核心。

---

## 4. 配置文件设计

示例：`expert_eoh/config/bp_online.yaml`

```yaml
project:
  name: expert_eoh_bp_online
  problem: bp_online
  output_dir: ./expert_eoh/results/bp_online_run1

# 桥接原 EoH evaluator 的配置。
# 实际路径请 Codex 检查仓库后调整。
evaluator:
  mode: subprocess  # subprocess 或 import
  eval_dir: ./examples/bp_online/evaluation
  heuristic_filename: heuristic.py
  run_command: python runEval.py
  objective_direction: minimize
  function_name: heuristic

llm:
  provider: openai_compatible
  model: gpt-5.5-thinking
  api_base: ${OPENAI_API_BASE}
  api_key: ${OPENAI_API_KEY}
  temperature: 0.7
  max_retries: 3
  timeout: 120

evolution:
  pop_size: 10
  generations: 20
  offspring_per_generation: 10
  parent_selection: rank
  survival_selection: greedy
  deduplicate_code: true

bootstrap:
  use_baseline_seed: true
  seed_path: ./expert_eoh/seeds/bp_online_baseline.json

agent:
  max_steps_per_offspring: 6
  max_repairs_per_candidate: 2
  enable_dynamic_skill_selection: true

memory:
  enable_compaction: true
  compact_every_generation: true
  compact_token_threshold: 60000
  keep_recent_events: 50

generalization:
  enable_audit: true
  enable_probes: true
  risk_threshold: 0.75
  risk_weight: 0.15
```

---

## 5. Baseline seed 设计

示例：`expert_eoh/seeds/bp_online_baseline.json`

```json
[
  {
    "id": "best_fit_seed",
    "algorithm": "Best-Fit baseline: place the current item into the feasible bin that leaves the smallest remaining capacity after placement.",
    "code": "def heuristic(item, bins):\n    import numpy as np\n    scores = item - bins\n    scores[bins < item] = -1e9\n    return scores"
  }
]
```

注意：

1. seed 代码必须完全符合当前任务 evaluator 的函数签名。
2. seed 中必须有 `algorithm` 和 `code` 两个字段。
3. 如果原 evaluator 要求特定 import 或函数名，请按原 evaluator 调整。
4. seed 不追求最强，但必须稳定、能跑、可解释。

---

## 6. Stage 1：Expert Bootstrap

### 6.1 目标

把 baseline seed 从“一个初始个体”升级为“专家起点”。

输入：

```text
baseline algorithm
baseline code
task interface
training instances / evaluator
```

输出：

```text
problem_card.json
candidate_store.json
expert_notebook.json
compact_context.json
```

### 6.2 `problem_card.json`

示例：

```json
{
  "problem_name": "Online Bin Packing",
  "heuristic_signature": "def heuristic(item, bins): return scores",
  "objective": "minimize number of used bins",
  "objective_direction": "minimize",
  "visible_information": [
    "current item size",
    "remaining capacities of currently open bins"
  ],
  "forbidden_information": [
    "future item sequence",
    "offline sorting of all items"
  ],
  "baseline_principle": "Best-Fit prefers the feasible bin with the tightest remaining capacity after placing the item.",
  "basic_design_principles": [
    "Preserve feasibility.",
    "Prefer normalized quantities over absolute thresholds.",
    "Avoid over-specialized hard-coded constants.",
    "Do not use information unavailable to the heuristic interface."
  ]
}
```

### 6.3 `candidate_store.json`

初始写入 baseline seed：

```json
{
  "best_fit_seed": {
    "id": "best_fit_seed",
    "parents": [],
    "algorithm": "Best-Fit baseline...",
    "code": "def heuristic(...): ...",
    "objective": 0.0,
    "mechanism_summary": "Best-Fit-like tight-fit preference.",
    "risk_score": 0.0,
    "contract_passed": true,
    "status": "seed",
    "generation": 0
  }
}
```

objective 应通过 evaluator 真实评估后填入。

### 6.4 `expert_notebook.json`

初始内容：

```json
{
  "validated_principles": [
    "Start from a simple baseline-like mechanism.",
    "Prefer small controlled corrections over complete rewrites.",
    "Do not destroy feasibility or baseline stability."
  ],
  "anti_patterns": [
    "Unexplained hard-coded constants",
    "Many unrelated nonlinear terms",
    "Branching rules tied to a narrow training distribution"
  ],
  "successful_repairs": [],
  "failed_hypotheses": [],
  "open_questions": [
    "Can a small correction term improve the baseline without damaging robustness?"
  ]
}
```

### 6.5 `compact_context.json`

初始内容：

```json
{
  "problem_card": {},
  "current_best": {
    "id": "best_fit_seed",
    "mechanism": "baseline-like tight-fit preference",
    "objective": null,
    "risk_score": 0.0
  },
  "recent_lessons": [],
  "active_design_rules": [
    "Preserve baseline core unless evidence suggests otherwise.",
    "Use controlled variation, not full rewrite."
  ],
  "next_recommended_focus": [
    "Try a small controlled variation of the baseline seed."
  ]
}
```

---

## 7. Stage 2：外层 EC Loop

外层结构：

```python
class ExpertEoHEngine:
    def run(self):
        population = self.bootstrap.initialize_population()

        for gen in range(self.config.generations):
            offspring = []

            for _ in range(self.config.offspring_per_generation):
                parent = self.selector.select_parent(population)

                candidate = self.harness.run(
                    parent=parent,
                    population=population,
                    memory=self.memory
                )

                if candidate is None:
                    continue

                objective = self.evaluator.evaluate(candidate.code)
                candidate.objective = objective

                self.memory.candidate_store.save(candidate)
                self.memory.session_log.log_eval(candidate)
                offspring.append(candidate)

            population = self.population_manager.update(
                population=population,
                offspring=offspring
            )

            self.memory.update_generation_summary(gen, population, offspring)
            self.memory.compact_if_needed()

        return population
```

### 7.1 parent selection

第一版用简单 rank selection 即可。

### 7.2 survival selection

第一版用 greedy top-N 即可。

如果是 minimize objective：保留 objective 最小的 N 个。

### 7.3 deduplication

候选代码完全相同则去重。第一版先做 exact string dedup，后续可以加 AST-level dedup。

---

## 8. Stage 3：Agent Harness Loop

### 8.1 核心 loop

```python
class AgentHarness:
    def run(self, parent, population, memory):
        state = AgentState(
            parent=parent,
            population_summary=memory.summarize_population(population),
            compact_context=memory.load_compact_context(),
            notebook=memory.load_notebook()
        )

        while state.step_count < self.config.max_steps_per_offspring:
            context = memory.build_context_for_agent(state)

            decision = self.llm.decide_next_skill(
                context=context,
                skill_metadata=self.skill_registry.metadata()
            )

            skill = self.skill_registry.load(decision.skill_name)
            result = self.tool_router.execute(
                skill=skill,
                decision=decision,
                state=state
            )

            state.update(result)
            memory.session_log.append(result)

            if state.needs_compaction():
                memory.compact()

            gate_decision = self.commit_gate.check(state)

            if gate_decision == "commit":
                return state.best_candidate()

            if gate_decision == "reject":
                return None

        return state.best_candidate_if_available()
```

### 8.2 agent 决策 JSON

每一步 LLM 必须输出 JSON：

```json
{
  "next_skill": "controlled_variation",
  "reason": "The parent mechanism is understood and we should try a small normalized correction to the baseline-like score.",
  "arguments": {
    "target_parent_id": "best_fit_seed",
    "variation_goal": "improve training performance without destroying baseline stability"
  }
}
```

禁止输出自由散文作为主响应。所有决策必须可解析。

---

## 9. Skill 系统

### 9.1 Skill Registry

启动时只把 skill metadata 放入 prompt：

```json
[
  {
    "name": "mechanism_decomposition",
    "description": "Use when the agent needs to understand how a parent heuristic works before modifying it."
  },
  {
    "name": "controlled_variation",
    "description": "Use when the agent has a hypothesis and wants to generate a small, controlled modification of the parent heuristic."
  },
  {
    "name": "contract_check",
    "description": "Use after code is generated to check function signature, imports, output shape, NaN/inf, and evaluator compatibility."
  },
  {
    "name": "repair",
    "description": "Use when candidate code fails contract checks, runtime checks, or generalization audits."
  },
  {
    "name": "generalization_audit",
    "description": "Use to detect overfitting risks such as hard-coded constants, excessive branching, non-normalized terms, and baseline regression."
  },
  {
    "name": "probe_synthesis",
    "description": "Use to create lightweight stress probes from existing instances and task invariances."
  },
  {
    "name": "context_compaction",
    "description": "Use when context grows too long or when a milestone should be summarized."
  },
  {
    "name": "notebook_update",
    "description": "Use after commit/reject to update expert design principles, anti-patterns, and open questions."
  }
]
```

只有当 agent 选择某个 skill 时，才加载该 skill 的完整 `SKILL.md`。

---

## 10. 核心 SKILL.md 模板

每个 `SKILL.md` 统一结构：

```markdown
---
name: skill_name
description: One-sentence description.
---

# Skill: skill_name

## Purpose

## When to use

## Inputs

## Outputs
Return JSON only.

## Procedure

## Failure modes

## JSON schema
```

---

## 11. Skill 1：mechanism_decomposition

路径：

```text
expert_eoh/skills/mechanism_decomposition/SKILL.md
```

内容：

```markdown
---
name: mechanism_decomposition
description: Use when the agent needs to understand how a parent heuristic works before modifying it.
---

# Skill: mechanism_decomposition

## Purpose
Decompose the parent heuristic into interpretable mechanism components, like a human expert reading an existing heuristic before modifying it.

## When to use
Use before generating a variation if the current parent has not been analyzed in the compact context.

## Inputs
- problem_card
- parent_algorithm
- parent_code
- parent_objective
- expert_notebook

## Outputs
Return JSON only:

{
  "dominant_mechanism": "...",
  "components": [
    {
      "name": "...",
      "code_region": "...",
      "intended_effect": "...",
      "possible_risk": "..."
    }
  ],
  "strengths": ["..."],
  "weaknesses": ["..."],
  "safe_to_modify": ["..."],
  "should_preserve": ["..."]
}

## Procedure
1. Read the parent thought and code.
2. Identify core scoring / decision terms.
3. Explain what each component is intended to do.
4. Identify possible brittleness or overfitting risk.
5. Separate components that should be preserved from components that can be modified.

## Failure modes
- If code cannot be understood, mark uncertainty and recommend contract_check or repair.
```

---

## 12. Skill 2：controlled_variation

路径：

```text
expert_eoh/skills/controlled_variation/SKILL.md
```

内容：

```markdown
---
name: controlled_variation
description: Use when the agent has a hypothesis and wants to generate a small, controlled modification of the parent heuristic.
---

# Skill: controlled_variation

## Purpose
Generate a new heuristic candidate by making a small expert-style modification to the parent. Avoid complete rewrites unless explicitly justified.

## When to use
Use after mechanism_decomposition or when the compact context contains a clear modification hypothesis.

## Inputs
- problem_card
- parent_algorithm
- parent_code
- parent_mechanism_summary
- expert_notebook
- active_design_rules

## Outputs
Return JSON only:

{
  "hypothesis": "...",
  "preserve": ["..."],
  "modify": ["..."],
  "remove": ["..."],
  "expected_effect": "...",
  "possible_side_effect": "...",
  "candidate_algorithm": "...",
  "candidate_code": "..."
}

## Procedure
1. Preserve the function signature exactly.
2. Preserve the baseline-like core unless there is a clear reason to alter it.
3. Modify at most one or two mechanisms.
4. Prefer normalized terms over raw absolute thresholds.
5. Avoid many unrelated constants.
6. Generate both natural-language algorithm description and executable code.

## Failure modes
- If the modification requires unavailable information, reject it.
- If generated code is too different from parent without justification, flag for audit.
```

---

## 13. Skill 3：contract_check

路径：

```text
expert_eoh/skills/contract_check/SKILL.md
```

内容：

```markdown
---
name: contract_check
description: Use after code is generated to check function signature, imports, output shape, NaN/inf, and evaluator compatibility.
---

# Skill: contract_check

## Purpose
Check whether candidate code satisfies the task interface and can be executed by the evaluator.

## When to use
Use after candidate generation or repair.

## Inputs
- problem_card
- candidate_code
- expected_function_name
- expected_signature
- optional mini_test_inputs

## Outputs
Return JSON only:

{
  "passed": true,
  "errors": [],
  "warnings": [],
  "recommended_next_skill": "commit"
}

or

{
  "passed": false,
  "errors": [
    {
      "type": "signature_error | syntax_error | runtime_error | output_shape_error | nan_inf_error | forbidden_information",
      "detail": "..."
    }
  ],
  "warnings": ["..."],
  "recommended_next_skill": "repair"
}

## Procedure
1. Check syntax.
2. Check function name.
3. Check input parameters.
4. Check imports.
5. Run small dummy inputs if available.
6. Check output type, shape, NaN/inf.
7. Check forbidden information usage if detectable.

## Failure modes
- If unable to execute, return passed=false with runtime_error.
```

Implementation note: `contract_check` can be partly deterministic Python code, not only LLM. Use AST parsing and small dummy evaluations where possible.

---

## 14. Skill 4：repair

路径：

```text
expert_eoh/skills/repair/SKILL.md
```

内容：

```markdown
---
name: repair
description: Use when candidate code fails contract checks, runtime checks, or generalization audits.
---

# Skill: repair

## Purpose
Repair a candidate heuristic based on precise failure reports, while preserving its intended mechanism when possible.

## When to use
Use after contract_check, mini_eval, or generalization_audit reports a fixable problem.

## Inputs
- problem_card
- candidate_algorithm
- candidate_code
- failure_report
- parent_code
- expert_notebook

## Outputs
Return JSON only:

{
  "repair_reason": "...",
  "preserved_mechanism": "...",
  "changed_parts": ["..."],
  "repaired_algorithm": "...",
  "repaired_code": "...",
  "expected_fix": "..."
}

## Procedure
1. Identify the exact failure type.
2. Keep the function signature unchanged.
3. Fix the smallest necessary part.
4. Preserve the candidate's intended mechanism if possible.
5. If the mechanism itself caused the failure, simplify it or revert to baseline-like behavior.

## Failure modes
- If more than two repair attempts fail, recommend reject.
```

---

## 15. Skill 5：generalization_audit

路径：

```text
expert_eoh/skills/generalization_audit/SKILL.md
```

内容：

```markdown
---
name: generalization_audit
description: Use to detect overfitting risks such as hard-coded constants, excessive branching, non-normalized terms, and baseline regression.
---

# Skill: generalization_audit

## Purpose
Audit whether a candidate heuristic is likely to overfit the small training set.

## When to use
Use after candidate passes contract_check, before commit, especially if the candidate is much more complex than the parent.

## Inputs
- problem_card
- parent_code
- candidate_code
- candidate_algorithm
- expert_notebook
- optional mini_eval_result

## Outputs
Return JSON only:

{
  "risk_score": 0.0,
  "risk_factors": [
    {
      "type": "hard_constant | absolute_threshold | excessive_branching | non_normalized_term | unexplained_complexity | baseline_regression",
      "evidence": "...",
      "suggested_fix": "..."
    }
  ],
  "recommended_next_skill": "commit | repair | probe_synthesis | reject"
}

## Procedure
1. Identify numeric constants and thresholds.
2. Check whether constants are normalized by instance scale.
3. Check branching complexity.
4. Check whether the candidate destroys the baseline-like core without justification.
5. Check whether complexity is explained by a clear mechanism.
6. Assign risk_score from 0 to 1.

## Failure modes
- If uncertain, recommend probe_synthesis rather than reject.
```

---

## 16. Skill 6：probe_synthesis

路径：

```text
expert_eoh/skills/probe_synthesis/SKILL.md
```

内容：

```markdown
---
name: probe_synthesis
description: Use to create lightweight stress probes from existing instances and task invariances.
---

# Skill: probe_synthesis

## Purpose
Generate cheap stress probes to test heuristic stability without adding a large new training set.

## When to use
Use when generalization_audit is uncertain or when a candidate has high training performance but suspicious complexity.

## Inputs
- problem_card
- candidate_code
- parent_code
- available_training_instances_summary
- expert_notebook

## Outputs
Return JSON only:

{
  "probes": [
    {
      "probe_name": "...",
      "transformation": "...",
      "expected_property": "...",
      "cost": "low | medium",
      "reason": "..."
    }
  ]
}

## Procedure
1. Use only information available from existing training instances or problem invariances.
2. Prefer transformations such as scaling, permutation, small perturbation, or boundary cases.
3. Do not create a large new benchmark set.
4. Each probe must have an expected property.

## Failure modes
- If no safe probe can be defined, return empty list and explain.
```

---

## 17. Skill 7：context_compaction

路径：

```text
expert_eoh/skills/context_compaction/SKILL.md
```

内容：

```markdown
---
name: context_compaction
description: Use when context grows too long or when a milestone should be summarized.
---

# Skill: context_compaction

## Purpose
Compress raw agent traces, candidate histories, and notebook updates into a compact context for future LLM calls.

## When to use
Use at the end of each generation, when token budget is high, or after major commit/reject events.

## Inputs
- recent_session_events
- candidate_store_summary
- expert_notebook
- current_population_summary
- problem_card

## Outputs
Return JSON only:

{
  "current_best": {
    "id": "...",
    "mechanism": "...",
    "objective": null,
    "risk_score": null
  },
  "important_recent_events": ["..."],
  "design_lessons": ["..."],
  "anti_patterns": ["..."],
  "discarded_details": ["..."],
  "next_focus": ["..."]
}

## Procedure
1. Keep information needed for future design.
2. Remove syntax-only failures and duplicate details unless they reveal a reusable pattern.
3. Preserve current elites and validated design rules.
4. Preserve open questions and recommended next actions.

## Failure modes
- If uncertain whether to discard an item, keep a short summary.
```

---

## 18. Memory 管理细节

### 18.1 session_log.jsonl

每个事件一行 JSON：

```json
{"t": 1, "event": "bootstrap_loaded_seed", "candidate_id": "best_fit_seed"}
{"t": 2, "event": "skill_called", "skill": "mechanism_decomposition", "parent": "best_fit_seed"}
{"t": 3, "event": "candidate_generated", "candidate_id": "h_001", "parent": "best_fit_seed"}
{"t": 4, "event": "contract_check", "candidate_id": "h_001", "passed": false}
{"t": 5, "event": "repair", "from": "h_001", "to": "h_002"}
{"t": 6, "event": "formal_eval", "candidate_id": "h_002", "objective": 0.512}
```

### 18.2 candidate_store.json

保存完整 candidate 信息。

### 18.3 expert_notebook.json

保存专家经验，不保存流水账。

### 18.4 compact_context.json

每次 prompt 的主要上下文来源。

---

## 19. Evaluator Bridge

因为 EoH examples 往往要求把 heuristic 写入 `heuristic.py` 后运行 `runEval.py`，第一版建议采用 subprocess 方式，最稳。

伪代码：

```python
class SubprocessEvaluatorBridge:
    def __init__(self, eval_dir, heuristic_filename, run_command, timeout=300):
        self.eval_dir = Path(eval_dir)
        self.heuristic_path = self.eval_dir / heuristic_filename
        self.run_command = run_command
        self.timeout = timeout

    def evaluate(self, code: str) -> float:
        self.heuristic_path.write_text(code, encoding="utf-8")

        result = subprocess.run(
            self.run_command.split(),
            cwd=str(self.eval_dir),
            capture_output=True,
            text=True,
            timeout=self.timeout
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        return self.parse_objective(result.stdout)
```

`parse_objective` 需要根据具体 evaluator 输出格式实现。Codex 应先检查对应 example 的 `runEval.py` 输出。

---

## 20. Commit Gate

一个 candidate 可以进入正式 evaluator，需要满足：

```text
1. contract_check passed
2. 不是重复代码
3. repair 次数未超过上限
4. risk_score 未超过阈值，或者已经经过 repair
5. mini-eval 没有明显劣于 parent/baseline
```

伪代码：

```python
class CommitGate:
    def check(self, state):
        if state.current_candidate is None:
            return "continue"

        if not state.contract_passed:
            if state.repair_count < self.max_repairs:
                return "repair"
            return "reject"

        if state.risk_score is not None and state.risk_score > self.risk_threshold:
            if state.repair_count < self.max_repairs:
                return "repair"
            return "reject"

        if state.mini_eval_score is not None:
            if state.too_much_worse_than_parent():
                return "reject"

        return "commit"
```

注意：

```text
Commit Gate 不是正式 fitness。
正式 fitness 仍然由 EoH evaluator 给出。
```

---

## 21. LLM JSON 调用

所有 agent 决策和 skill 输出都必须通过 JSON schema 约束。

建议实现：

```text
llm/json_call.py
```

功能：

```text
1. 构造 prompt
2. 调用 LLM
3. 提取 JSON
4. schema 校验
5. 失败重试
6. 最多 max_retries 次
```

错误处理：

```text
JSON 不合法 → 要求模型修复 JSON
字段缺失 → 自动补 retry prompt
代码块缺失 → retry
超过重试次数 → 返回失败，由 agent state 记录
```

---

## 22. 实现路线

由于用户已经复现过 EoH，所以不需要完整 V0。只做 smoke test。

### Step 0：Smoke test

目标：新工程能调用原 evaluator。

任务：

```text
1. 创建 expert_eoh/
2. 读取 baseline seed
3. 写入 evaluator 的 heuristic.py
4. 跑 runEval.py
5. 解析 objective
```

### Step 1：Expert Bootstrap

实现：

```text
bootstrap/baseline_loader.py
bootstrap/problem_card.py
bootstrap/expert_bootstrap.py
memory/candidate_store.py
memory/expert_notebook.py
memory/compact_context.py
```

输出：

```text
problem_card.json
candidate_store.json
expert_notebook.json
compact_context.json
```

### Step 2：4 个基础 skills

实现：

```text
mechanism_decomposition
controlled_variation
contract_check
repair
```

目标：agent 从 baseline parent 生成一个能跑 candidate。

### Step 3：Agent Harness Loop

实现：

```text
harness/agent_loop.py
harness/agent_state.py
harness/skill_registry.py
harness/tool_router.py
harness/commit_gate.py
```

目标：同一个 candidate 可以经历：

```text
choose_skill → execute → observe → choose_next_skill
```

### Step 4：外层 EC loop

实现：

```text
core/engine.py
core/population.py
core/selection.py
core/result_store.py
```

目标：跑多代，每代多个 offspring。

### Step 5：Memory + Compaction

实现：

```text
session_log.jsonl
candidate_store.json
expert_notebook.json
compact_context.json
context_compaction skill
```

目标：每代结束自动 compact。

### Step 6：Generalization Audit / Probe

实现：

```text
generalization_audit
probe_synthesis
risk_score
commit gate risk logic
```

目标：减少明显过拟合、复杂、脆弱的候选进入正式选择。

---

## 23. 实验版本和消融

最终至少要能跑这些版本：

```text
1. EoH original
2. EoH + baseline seed
3. Expert Bootstrap + one-shot agent
4. Expert Bootstrap + dynamic agent loop
5. + memory / compaction
6. Full method: + generalization audit / probes
```

比较指标：

```text
training objective
unseen test objective
train-test gap
invalid code rate
repair success rate
token cost
每代 best-so-far
risk_score 与 unseen test performance 的相关性
```

---

## 24. Codex 执行注意事项

请 Codex 按以下原则实现：

1. **不要破坏原 EoH 目录。** 只在 `expert_eoh/` 写新代码，必要时通过 import 或 subprocess 调用原 evaluator。
2. **先 inspect 仓库实际路径。** 如果 README 或 examples 的路径与本文不同，以实际仓库为准。
3. **先做 smoke test。** baseline seed 必须能被 evaluator 评估。
4. **所有 LLM 输出都要求 JSON。** 不要让 agent 返回不可解析散文。
5. **skills 使用 progressive disclosure。** 启动时只加载 skill metadata；选择 skill 后再加载完整 `SKILL.md`。
6. **保存所有 trace。** 每次 skill 调用、LLM 响应、candidate 生成、repair、eval 都写 session log。
7. **context compaction 必须可复现。** compact 前后的内容都要落盘。
8. **不要一次性做太多。** 按 Step 0 → Step 6 实现，每一步都要有可运行测试。

---

## 25. 最终一句话

本项目不是“给 EoH 增加一个 agent operator”，而是：

```text
在 EoH 旁边重写一个 baseline-seeded expert-agent harness：
保留 EoH 的任务接口和 evaluator，
淘汰固定 prompt operators，
用 baseline seed 冷启动，
用 expert notebook 作为记忆，
用 context compaction 控制长程上下文，
用 skill-based agent loop 动态分析、生成、检查、修复、提交候选，
并通过 generalization audit / lightweight probes 抑制少样本过拟合。
```

