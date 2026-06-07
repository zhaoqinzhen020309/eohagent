# eoh.py - SEC V6 Framework Main Controller
import numpy as np
import json
import random
import time
import os
import sys

from .eoh_interface_EC import InterfaceEC


class EVOL:
    """SEC V6: Self-adaptive Evolution with Collaborative LLMs"""

    def __init__(self, paras, problem=None, select=None, manage=None):
        """Initialize SEC V6 with dual-LLM configuration

        Args:
            paras: Parameter object containing all configuration
            problem: (Optional) Problem instance, will use paras.problem if not provided
            select: (Optional) Selection strategy, will create placeholder if not provided
            manage: (Optional) Management strategy, will create placeholder if not provided
        """

        # === 从 paras 对象获取所有配置 ===
        try:
            # 问题实例 - 优先使用传入的，否则从paras获取
            self.prob = problem if problem is not None else paras.problem

            # 选择和管理策略
            # 优先级：传入参数 > paras属性 > 占位符
            if select is not None:
                self.select = select
            elif hasattr(paras, 'select') and paras.select is not None:
                self.select = paras.select
            else:
                print("Warning: Using placeholder for Select class.")
                self.select = self._create_placeholder_select()

            if manage is not None:
                self.manage = manage
            elif hasattr(paras, 'manage') and paras.manage is not None:
                self.manage = paras.manage
            else:
                print("Warning: Using placeholder for Manage class.")
                self.manage = self._create_placeholder_manage()

            # LLM 1 配置 (Engineer/Worker)
            self.llm_use_local_1 = paras.llm_use_local_1
            self.llm_local_url_1 = paras.llm_local_url_1
            self.api_endpoint_1 = paras.llm_api_endpoint_1
            self.api_key_1 = paras.llm_api_key_1
            self.llm_model_1 = paras.llm_model_1

            # LLM 2 配置 (Strategist)
            self.llm_use_local_2 = paras.llm_use_local_2
            self.llm_local_url_2 = paras.llm_local_url_2
            self.api_endpoint_2 = paras.llm_api_endpoint_2
            self.api_key_2 = paras.llm_api_key_2
            self.llm_model_2 = paras.llm_model_2

            # EC & 实验参数
            self.pop_size = paras.ec_pop_size
            self.n_pop = paras.ec_n_pop
            self.p = paras.ec_selection_num  # 父代样本数 (给 LLM 2)
            self.r = paras.ec_portfolio_size  # 算子组合大小
            self.debug_mode = paras.exp_debug_mode
            self.output_path = paras.exp_output_path
            self.exp_n_proc = paras.exp_n_proc
            self.timeout = paras.eva_timeout

            # 加载/继续运行参数
            self.use_seed = getattr(paras, 'exp_use_seed', False)
            self.seed_path = getattr(paras, 'exp_seed_path', '')
            self.load_pop = paras.exp_use_continue
            self.load_pop_path = paras.exp_continue_path
            self.load_pop_id = paras.exp_continue_id

        except AttributeError as e:
            print(f"Error accessing parameters from 'paras' object: {e}")
            print("Please ensure Paras class definition includes all required attributes.")
            sys.exit(1)

        print("=" * 70)
        print("SEC V6 Framework Initialized")
        print("=" * 70)
        print(f"  * LLM 1 (Engineer/Worker): {self.llm_model_1}")
        print(f"  * LLM 2 (Strategist): {self.llm_model_2}")
        print(f"  * Portfolio Size (r): {self.r}")
        print(f"  * Strategist Samples (p): {self.p}")
        print(f"  * Population Size: {self.pop_size}")
        print(f"  * Generations: {self.n_pop}")
        print("=" * 70)

        random.seed(2024)

        # 确保输出目录存在
        try:
            os.makedirs(os.path.join(self.output_path, "results", "pops"), exist_ok=True)
            os.makedirs(os.path.join(self.output_path, "results", "pops_best"), exist_ok=True)
        except OSError as e:
            print(f"Error creating output directories: {e}")
            sys.exit(1)

    def _create_placeholder_select(self):
        """创建占位符选择器"""

        class PlaceholderSelect:
            def ranked_selection(self, pop, num):
                if not pop:
                    return []
                sorted_pop = sorted([p for p in pop if p.get('objective') is not None],
                                    key=lambda x: x['objective'])
                return sorted_pop[:min(num, len(sorted_pop))]

        return PlaceholderSelect()

    def _create_placeholder_manage(self):
        """创建支持多样性的管理器 - 锦标赛选择 + 精英保留"""

        class DiversityAwareManage:
            def population_management(self, pop, target_size):
                """
                种群管理策略：精英保留 + 锦标赛选择

                Args:
                    pop: 当前种群（包含旧种群和新后代）
                    target_size: 目标种群大小

                Returns:
                    下一代种群
                """
                if not pop:
                    return []

                # 过滤掉无效个体
                valid_pop = [p for p in pop if p.get('objective') is not None]

                # 无论大小是否匹配，都要排序！确保最佳个体在第一位
                sorted_pop = sorted(valid_pop, key=lambda x: x['objective'])

                if len(sorted_pop) <= target_size:
                    return sorted_pop

                # 1. 精英保留：确保最好的个体存活（保留前20%）
                elite_size = max(1, int(target_size * 0.2))
                next_gen = sorted_pop[:elite_size].copy()

                print(f"    [Population Management] Elite preservation: keeping top {elite_size} individuals")

                # 2. 锦标赛选择：从剩余个体中选择，给探索性个体机会
                remaining = sorted_pop[elite_size:]
                tournament_size = 3  # 每次从3个个体中选最好的

                selected_count = 0
                while len(next_gen) < target_size and remaining:
                    # 从剩余个体中随机选择 tournament_size 个进行锦标赛
                    actual_tournament_size = min(tournament_size, len(remaining))
                    tournament = random.sample(remaining, actual_tournament_size)

                    # 选择锦标赛中适应度最好的
                    winner = min(tournament, key=lambda x: x['objective'])
                    next_gen.append(winner)
                    remaining.remove(winner)
                    selected_count += 1

                print(
                    f"    [Population Management] Tournament selection: selected {selected_count} additional individuals")
                print(f"    [Population Management] Final population size: {len(next_gen)}")

                return next_gen[:target_size]

        return DiversityAwareManage()

    def add2pop(self, population, offspring_list):
        """将有效的、非重复的后代添加到种群中"""
        added_count = 0
        if not isinstance(offspring_list, list):
            print("[Warning] add2pop received non-list input for offspring.")
            return 0

        existing_codes = {ind.get('code') for ind in population if ind.get('code')}

        for off in offspring_list:
            if off is None or off.get('objective') is None or not off.get('code'):
                if self.debug_mode:
                    print("Skipping invalid offspring (None or missing objective/code).")
                continue

            off_code = off.get('code')
            if off_code in existing_codes:
                if self.debug_mode:
                    print(f"Skipping duplicate offspring code (Fitness: {off.get('objective'):.5f}).")
                continue

            # 检查 objective 是否重复
            is_obj_duplicate = False
            for ind in population:
                if ind.get('objective') is not None and np.isclose(ind.get('objective'), off.get('objective'),
                                                                   atol=1e-6):
                    is_obj_duplicate = True
                    break
            if is_obj_duplicate:
                if self.debug_mode:
                    print(f"Skipping offspring with duplicate objective ({off.get('objective'):.5f}).")
                continue

            population.append(off)
            existing_codes.add(off_code)
            added_count += 1

        return added_count

    def _build_meta_prompt(self, generation, stagnation_count, selected_parents, last_feedback=None):
        """
        (重构后 - 精简版)
        构建 Strategist LLM 2 的 Prompt。
        专注于设计抽象的 EC 算子 (P_draft)，不包含问题特定约束。
        根据问题类型自动选择对应的prompt模板。
        """

        # 0. 识别问题类型
        problem_type = self._identify_problem_type()

        # 1. 构建父代样本字符串
        p = len(selected_parents)
        parent_samples_text = f"**# Current Population Samples (p={p})**:\n\n"
        parent_samples_text += f"**IMPORTANT: Lower fitness is better. Individual 1 has the smallest fitness value and performs best.**\n\n"
        if not selected_parents:
            parent_samples_text += "  (No valid parents selected)\n"
        else:
            for i, parent in enumerate(selected_parents):
                obj = parent.get('objective', 'N/A')
                thought = parent.get('algorithm', 'N/A')
                full_code = parent.get('code', '# N/A')

                parent_samples_text += f"{i + 1}. **Individual {i + 1} (Fitness: {obj if isinstance(obj, (int, float, np.number)) else 'N/A'})**:\n"
                parent_samples_text += f"   * Thought: {{ {thought} }}\n"
                parent_samples_text += f"   * Full Code:\n```python\n{full_code}\n```\n"

        # 2. 构建反馈字符串
        feedback_text = "**# Last Generation Feedback**:\n  (None)\n"
        if last_feedback:
            feedback_text = f"**# Last Generation Feedback**:\n{last_feedback}\n"

        # 3. 根据问题类型选择对应的prompt模板
        if problem_type == "TSP":
            return self._build_tsp_meta_prompt(generation, stagnation_count, p, parent_samples_text, feedback_text)
        elif problem_type == "FSSP":
            return self._build_fssp_meta_prompt(generation, stagnation_count, p, parent_samples_text, feedback_text)
        else:  # Default: Bin Packing
            return self._build_binpacking_meta_prompt(generation, stagnation_count, p, parent_samples_text,
                                                      feedback_text)

    def _identify_problem_type(self):
        """识别问题类型"""
        problem_class_name = self.prob.__class__.__name__
        if "TSP" in problem_class_name.upper():
            return "TSP"
        elif "JSSP" in problem_class_name.upper() or "FSSP" in problem_class_name.upper():
            return "FSSP"
        else:
            return "BinPacking"

    def _build_binpacking_meta_prompt(self, generation, stagnation_count, p, parent_samples_text, feedback_text):
        """构建Bin Packing问题的Strategist Prompt"""
        meta_prompt = f"""You are a strategy advisor for an evolutionary algorithm. Your task is to generate evolutionary guidance for Worker LLM to help it learn from parents and innovate.

**# Background**

Worker LLM is evolving a scoring function for the "online bin packing" problem.

Function signature: def score(item, bins) -> scores
Code environment: Can only use Python standard library and NumPy

**# Your Task**

Observe the complete code of the 5 parents below and generate 2 evolutionary guidance instructions (P_drafts) for Worker.

Worker will see the complete code of all parents. Your P_draft provides evolutionary **direction and ideas**, not specific execution steps.

**Important: Lower fitness is better.** Individual 1 has the smallest fitness and performs best.

**# Current Population State**

Generation: {generation}, Stagnation: {stagnation_count}

**# Parent Samples (p={p})**:

**IMPORTANT: Lower fitness is better. Individual 1 has the smallest fitness value and performs best.**

{parent_samples_text}

{feedback_text}

**# How to Generate Effective P_draft**

**Critical Understanding of P_draft Usage**

The Worker LLM receives a prompt structured like this:
```
Firstly, {{YOUR_P_DRAFT}}
Secondly, analyze all parent heuristics in detail...
Thirdly, implement your new heuristic in code...
```

Your P_draft will be inserted into the "Firstly" section, so it MUST be a simple, clear, executable task description.

**Good P_draft Examples (short, open, action-oriented)**:

1. "identify the common approach and create a variation"
2. "combine different techniques from multiple parents"
3. "explore a fundamentally different scoring structure"

**Bad P_draft Examples (specific, constrained)**:

1. BAD: "identify patterns from top-performing parents and refine them"
   - Problem: "top-performing" only focuses on best parents, ignores innovations in others
   - Problem: "refine" is vague, Worker doesn't know how to refine

2. BAD: "create a balanced scoring function using geometric mean of utilization and penalty factors"
   - Problem: Specifies "geometric mean", limits Worker's creativity
   - Problem: Multiple sub-tasks (balanced, geometric mean, multiple factors), too complex

**Key Principles**:

- Keep it brief (under 15 words)
- Keep it open (don't specify HOW to do it)
- Use simple verbs ("combine", "create", "explore", "identify")
- Tell Worker WHAT to do, not HOW to do it
- Leave room for Worker's creativity

**AVOID in Your P_draft (with explanations)**:

- "balanced", "refined", "sophisticated", "optimized"
  Problem: Vague adjectives. What is "balanced"? Worker cannot judge.
  Instead: Use concrete verbs like "combine", "merge", "create"

- "top-performing", "best", "optimal"
  Problem: Limits scope to only best parents, ignoring innovations in others.
  Instead: Say "from parents" or "across all parents"

- Specific method names (e.g., "geometric mean", "cubic interpolation")
  Problem: Constrains Worker to exact techniques, no creativity.
  Instead: Say "different mathematical form" or "new approach"

- Multiple sub-tasks in one P_draft
  Problem: Too complex for Worker to execute in one step.
  Instead: Keep one clear task per P_draft

**# Two Strategy Types**

**1. Exploitation**: Guide Worker to learn from parents. Keep it simple - what to look for and what to do with it.

**2. Exploration**: Guide Worker to try new directions. Keep it simple - what to move away from and what new direction to try.

**Constraint**: Do not suggest methods requiring cross-call information.

**# Output Format**

Return ONLY valid JSON:

```json
{{
  "portfolio": [
    {{
      "strategy_type": "Exploitation",
      "P_draft": "[Your suggestion - keep under 15 words]"
    }},
    {{
      "strategy_type": "Exploration",
      "P_draft": "[Your suggestion - keep under 15 words]"
    }}
  ]
}}
```

Requirements:
1. Valid JSON with no syntax errors
2. Exactly {self.r} strategies in portfolio
3. Each P_draft should be SHORT and OPEN
4. Output ONLY JSON"""

        return meta_prompt

    def _build_tsp_meta_prompt(self, generation, stagnation_count, p, parent_samples_text, feedback_text):
        """构建TSP问题的Strategist Prompt"""
        meta_prompt = f"""You are a strategy advisor for an evolutionary algorithm. Your task is to generate evolutionary guidance for Worker LLM to help it learn from parents and innovate.

**# Background**

Worker LLM is evolving an edge distance update function for the "TSP Guided Local Search (GLS)" problem.

Function signature: def update_edge_distance(edge_distance, local_opt_tour, edge_n_used) -> updated_edge_distance
Code environment: Can only use Python standard library and NumPy

**# Your Task**

Observe the complete code of the 5 parents below and generate 2 evolutionary guidance instructions (P_drafts) for Worker.

Worker will see the complete code of all parents. Your P_draft provides evolutionary **direction and ideas**, not specific execution steps.

**IMPORTANT: Lower fitness is better.** (Individual 1 has the lowest fitness).
**Your task is to find innovative ideas from ALL parents. A lower-ranked parent (e.g., Individual 4 or 5) might have a creative idea that is not yet fully optimized. Do not just focus on Individual 1.**

**# Current Population State**

Generation: {generation}, Stagnation: {stagnation_count}

**# Parent Samples (p={p})**:

**IMPORTANT: Lower fitness is better.** (Individual 1 has the lowest fitness).
**Your task is to find innovative ideas from ALL parents. A lower-ranked parent (e.g., Individual 4 or 5) might have a creative idea that is not yet fully optimized. Do not just focus on Individual 1.**

{parent_samples_text}

{feedback_text}

**# How to Generate Effective P_draft**

**Critical Understanding of P_draft Usage**

The Worker LLM receives a prompt structured like this:
```
Firstly, {{YOUR_P_DRAFT}}
Secondly, analyze all parent heuristics in detail...
Thirdly, implement your new heuristic in code...
```

Your P_draft will be inserted into the "Firstly" section, so it MUST be a simple, clear, executable task description.

**Good P_draft Examples (short, open, action-oriented)**:

1. "identify the common approach and create a variation"
2. "combine different techniques from multiple parents"
3. "explore a fundamentally different update strategy"

**Bad P_draft Examples (specific, constrained)**:

1. BAD: "identify patterns from top-performing parents and refine them"
   (Also BAD: "best-performing parent", "highest-ranked parent" - same problem)
   - Problem: Focusing on only the best parent(s) ignores innovations in others
   - Problem: "refine" is vague, Worker doesn't know how to refine
   - Better: "identify patterns across ALL parents"

2. BAD: "create a balanced function using exponential decay of edge usage"
   - Problem: Specifies "exponential decay", limits Worker's creativity
   - Problem: Multiple sub-tasks (balanced, exponential decay), too complex

**3. CRITICAL FAILURE: "Combine the most effective penalty patterns" or "Combine geometric and harmonic mean"**
   - **Problem: This is a TRAP.** In real runs, repeating this command led to "snowballing complexity" (e.g., 4-5 multiplicative factors) and immediate stagnation. **This is NOT creative.** It creates overly complex code that cannot be improved.
   - **Better: "isolate one parent's idea and simplify it" OR "create a new mathematical form for the decay factor"**

**Key Principles**:

- Keep it brief (under 15 words)
- Keep it open (don't specify HOW to do it)
- Use simple verbs ("combine", "create", "explore", "identify")
- Tell Worker WHAT to do, not HOW to do it
- Leave room for Worker's creativity

**AVOID in Your P_draft (with explanations)**:

- "balanced", "refined", "sophisticated", "optimized"
  Problem: Vague adjectives. What is "balanced"? Worker cannot judge.
  Instead: Use concrete verbs like "combine", "merge", "create"

- "top-performing", "best", "optimal", "most effective"
  Problem: Limits scope to only best parents, ignoring innovations in others.
  Instead: Say "from parents" or "across all parents"

- Specific method names (e.g., "exponential decay", "sigmoid function", "normalized frequency")
  Problem: Constrains Worker to exact techniques, no creativity.
  Instead: Say "different mathematical form" or "new approach"

- Repeated patterns across generations
  Problem: If you used "combine X from parents" last time, try a different verb or focus.
  Instead: Vary your language: "identify", "merge", "extract", "isolate", "blend"

- Multiple sub-tasks in one P_draft
  Problem: Too complex for Worker to execute in one step.
  Instead: Keep one clear task per P_draft

**# Two Strategy Types**

**1. Exploitation**: Guide Worker to learn from parents. Keep it simple - what to look for and what to do with it.

**2. Exploration**: Guide Worker to try new directions. Keep it simple - what to move away from and what new direction to try.

**Constraint**: Do not suggest methods requiring cross-call information (e.g., historical records, state memory).

**# Output Format**

Return ONLY valid JSON:

```json
{{
  "portfolio": [
    {{
      "strategy_type": "Exploitation",
      "P_draft": "[Your suggestion - keep under 15 words]"
    }},
    {{
      "strategy_type": "Exploration",
      "P_draft": "[Your suggestion - keep under 15 words]"
    }}
  ]
}}
```

Requirements:
1. Valid JSON with no syntax errors
2. Exactly {self.r} strategies in portfolio
3. Each P_draft should be SHORT and OPEN
4. Output ONLY JSON"""

        return meta_prompt

    def _build_fssp_meta_prompt(self, generation, stagnation_count, p, parent_samples_text, feedback_text):
        """构建FSSP问题的Strategist Prompt"""
        meta_prompt = f"""You are a strategy advisor for an evolutionary algorithm. Your task is to generate evolutionary guidance for Worker LLM to help it learn from parents and innovate.

**# Background**

Worker LLM is evolving a time matrix update and job selection function for the "FSSP Guided Local Search" problem.

Function signature: def get_matrix_and_jobs(current_sequence, time_matrix, m, n) -> new_matrix, perturb_jobs
Code environment: Can only use Python standard library and NumPy

**# Your Task**

Observe the complete code of the 5 parents below and generate 2 evolutionary guidance instructions (P_drafts) for Worker.

Worker will see the complete code of all parents. Your P_draft provides evolutionary **direction and ideas**, not specific execution steps.

**IMPORTANT: Lower fitness is better.** (Individual 1 has the lowest fitness).
**Your task is to find innovative ideas from ALL parents. A lower-ranked parent (e.g., Individual 4 or 5) might have a creative idea that is not yet fully optimized. Do not just focus on Individual 1.**

**# Current Population State**

Generation: {generation}, Stagnation: {stagnation_count}

**# Parent Samples (p={p})**:

**IMPORTANT: Lower fitness is better.** (Individual 1 has the lowest fitness).
**Your task is to find innovative ideas from ALL parents. A lower-ranked parent (e.g., Individual 4 or 5) might have a creative idea that is not yet fully optimized. Do not just focus on Individual 1.**

{parent_samples_text}

{feedback_text}

**# How to Generate Effective P_draft**

**Critical Understanding of P_draft Usage**

The Worker LLM receives a prompt structured like this:
```
Firstly, {{YOUR_P_DRAFT}}
Secondly, analyze all parent heuristics in detail...
Thirdly, implement your new heuristic in code...
```

Your P_draft will be inserted into the "Firstly" section, so it MUST be a simple, clear, executable task description.

**Good P_draft Examples (short, open, action-oriented)**:

1. "identify the common approach and create a variation"
2. "combine different techniques from multiple parents"
3. "explore a fundamentally different matrix update strategy"

**Bad P_draft Examples (specific, constrained)**:

1. BAD: "identify patterns from top-performing parents and refine them"
   (Also BAD: "best-performing parent", "highest-ranked parent" - same problem)
   - Problem: Focusing on only the best parent(s) ignores innovations in others
   - Problem: "refine" is vague, Worker doesn't know how to refine
   - Better: "identify patterns across ALL parents"

2. BAD: "create a balanced function using exponential scaling of job criticality"
   - Problem: Specifies "exponential scaling", limits Worker's creativity
   - Problem: Multiple sub-tasks (balanced, exponential scaling), too complex

**3. CRITICAL FAILURE: "Combine the most effective penalty patterns" or "Combine geometric and harmonic mean"**
   - **Problem: This is a TRAP.** In real runs, repeating this command led to "snowballing complexity" (e.g., 4-5 multiplicative factors) and immediate stagnation. **This is NOT creative.** It creates overly complex code that cannot be improved.
   - **Better: "isolate one parent's idea and simplify it" OR "create a new mathematical form for the scaling factor"**

**Key Principles**:

- Keep it brief (under 15 words)
- Keep it open (don't specify HOW to do it)
- Use simple verbs ("combine", "create", "explore", "identify")
- Tell Worker WHAT to do, not HOW to do it
- Leave room for Worker's creativity

**AVOID in Your P_draft (with explanations)**:

- "balanced", "refined", "sophisticated", "optimized"
  Problem: Vague adjectives. What is "balanced"? Worker cannot judge.
  Instead: Use concrete verbs like "combine", "merge", "create"

- "top-performing", "best", "optimal", "most effective"
  Problem: Limits scope to only best parents, ignoring innovations in others.
  Instead: Say "from parents" or "across all parents"

- Specific method names (e.g., "exponential decay", "sigmoid function", "normalized frequency")
  Problem: Constrains Worker to exact techniques, no creativity.
  Instead: Say "different mathematical form" or "new approach"

- Repeated patterns across generations
  Problem: If you used "combine X from parents" last time, try a different verb or focus.
  Instead: Vary your language: "identify", "merge", "extract", "isolate", "blend"

- Multiple sub-tasks in one P_draft
  Problem: Too complex for Worker to execute in one step.
  Instead: Keep one clear task per P_draft

**# Two Strategy Types**

**1. Exploitation**: Guide Worker to learn from parents. Keep it simple - what to look for and what to do with it.

**2. Exploration**: Guide Worker to try new directions. Keep it simple - what to move away from and what new direction to try.

**Constraint**: Do not suggest methods requiring cross-call information (e.g., historical records, state memory).

**# Output Format**

Return ONLY valid JSON:

```json
{{
  "portfolio": [
    {{
      "strategy_type": "Exploitation",
      "P_draft": "[Your suggestion - keep under 15 words]"
    }},
    {{
      "strategy_type": "Exploration",
      "P_draft": "[Your suggestion - keep under 15 words]"
    }}
  ]
}}
```

Requirements:
1. Valid JSON with no syntax errors
2. Exactly {self.r} strategies in portfolio
3. Each P_draft should be SHORT and OPEN
4. Output ONLY JSON"""

        return meta_prompt

    def _get_fallback_m2_prompt(self, best_individual):
        """(重构后) 获取回退 Exploitation P_draft - 纯算子指令"""

        m2_p_draft = """identify the main algorithm parameters in the best-performing parent heuristic. Then, create a new heuristic that has different parameter settings. Focus on adjusting numerical constants, exponents, or weights while preserving the core mathematical structure."""

        return m2_p_draft

    def _get_fallback_explore_prompt(self):
        """(重构后) 获取回退 Exploration P_draft - 纯算子指令"""

        explore_p_draft = """create a new heuristic that has a totally different form from the given ones. Explore unconventional mathematical transformations or combinations not present in the current population."""

        return explore_p_draft

    def run(self):
        """SEC V6主循环"""
        print("\n" + "=" * 70)
        print("SEC V6 Evolution Start")
        print("=" * 70)

        time_start = time.time()

        # === 初始化 InterfaceEC ===
        try:
            interface_ec = InterfaceEC(
                pop_size=self.pop_size,
                # LLM 1 Configs
                api_endpoint_1=self.api_endpoint_1, api_key_1=self.api_key_1, llm_model_1=self.llm_model_1,
                llm_use_local_1=self.llm_use_local_1, llm_local_url_1=self.llm_local_url_1,
                # LLM 2 Configs
                api_endpoint_2=self.api_endpoint_2, api_key_2=self.api_key_2, llm_model_2=self.llm_model_2,
                llm_use_local_2=self.llm_use_local_2, llm_local_url_2=self.llm_local_url_2,
                # Other params
                debug_mode=self.debug_mode, interface_prob=self.prob, select=self.select,
                n_p=self.exp_n_proc, timeout=self.timeout,
                use_numba=False
            )
        except Exception as e:
            print(f"Error initializing InterfaceEC: {e}")
            sys.exit(1)

        # === 初始化种群 ===
        population = []
        n_start = 0
        last_feedback_to_strategist = None

        # 处理加载逻辑
        if self.load_pop:
            try:
                print(f"Loading population from {self.load_pop_path}")
                with open(self.load_pop_path) as file:
                    population = json.load(file)
                if not isinstance(population, list) or len(population) == 0:
                    raise ValueError("Loaded population is empty or invalid.")

                # 过滤无效个体并排序（无论大小是否匹配都要排序！）
                valid_population = [p for p in population if p.get('objective') is not None]
                if len(valid_population) == 0:
                    raise ValueError("Loaded population has no valid individuals with objectives.")

                # 按适应度排序（越小越好）
                population = sorted(valid_population, key=lambda x: x['objective'])

                # 如果大小不匹配，调整到目标大小
                if len(population) != self.pop_size:
                    print(
                        f"Warning: Loaded population size ({len(population)}) differs from target ({self.pop_size}). Adjusting...")
                    if len(population) > self.pop_size:
                        population = population[:self.pop_size]
                    else:
                        print(
                            f"Warning: Loaded population is smaller than target size. Will generate new individuals if needed.")

                print(
                    f"Initial population loaded! Size: {len(population)}, Best objective: {population[0]['objective']:.5f}")
                n_start = self.load_pop_id

                if not all(isinstance(ind, dict) and 'code' in ind and 'objective' in ind for ind in population):
                    print("[ERROR] Loaded population contains invalid individuals. Exiting.")
                    sys.exit(1)

            except FileNotFoundError:
                print(f"Error: Continue file not found at {self.load_pop_path}. Starting from scratch.")
                population = []
                n_start = 0
            except Exception as e:
                print(f"Error loading population: {e}. Starting from scratch.")
                population = []
                n_start = 0

        # 执行初始化
        if not population:
            attempts = 0
            max_attempts = self.pop_size * 10
            print("Creating initial population using i1 operator...")
            while len(population) < self.pop_size and attempts < max_attempts:
                if self.debug_mode:
                    print(f"Attempt {attempts + 1}, Current valid: {len(population)}/{self.pop_size}")
                offspring = interface_ec.get_one_i1_individual()
                if offspring:
                    added = self.add2pop(population, [offspring])
                    if added > 0:
                        if self.debug_mode:
                            print(f"  Success! Objective: {offspring['objective']:.5f}")
                    else:
                        if self.debug_mode:
                            print("  Duplicate or invalid offspring skipped by add2pop.")
                else:
                    if self.debug_mode:
                        print("  Failed to generate valid individual (returned None).")
                attempts += 1

            if len(population) < self.pop_size:
                print(
                    f"[ERROR] Failed to create initial population of size {self.pop_size} after {max_attempts} attempts.")
                sys.exit(1)

            print(f"Initial population created with {len(population)} individuals.")
            population = sorted(population, key=lambda x: x.get('objective', float('inf')))

            # 保存初始种群
            try:
                init_pop_filename = os.path.join(self.output_path, "results", "pops", "population_generation_0.json")
                with open(init_pop_filename, 'w') as f:
                    json.dump(population, f, indent=4)
                print(f"Initial population saved to {init_pop_filename}")
            except Exception as e:
                print(f"Warning: Failed to save initial population: {e}")

        # === 主世代循环 ===
        best_objective_history = [ind['objective'] for ind in population if ind.get('objective') is not None]
        stagnation_count = 0

        for pop in range(n_start, self.n_pop):
            print(f"\n{'=' * 70}")
            print(f"Generation {pop + 1} / {self.n_pop} - SEC V6")
            print(f"{'=' * 70}")

            # 设置当前代数（用于LLM交互记录）
            interface_ec.evol.current_generation = pop + 1

            # 1. 选择父代样本给 LLM 2
            selected_parents = []
            if population:
                try:
                    selected_parents = self.select.ranked_selection(population, self.p)
                    if len(selected_parents) < self.p:
                        print(f"Warning: Selected only {len(selected_parents)} parents (less than p={self.p}).")
                except Exception as e:
                    print(f"Error during parent selection: {e}. Proceeding without parent samples.")
            else:
                print("Warning: Population is empty, cannot select parents for Strategist.")

            # 2. 构建 Meta-Prompt
            # === ABLATION 3: 去除状态反馈 ===
            # 不传递generation、stagnation_count、last_feedback，让Strategist变成"无记忆"
            print(f"[ABLATION 3] Removing state feedback (generation={pop+1}, stagnation={stagnation_count})")
            meta_prompt = self._build_meta_prompt(0, 0, selected_parents, None)
            if self.debug_mode:
                print(f"\n[DEBUG] Meta-Prompt for LLM 2 (length: {len(meta_prompt)} chars):")
                print("=" * 80)
                print(meta_prompt)  # 完整输出
                print("=" * 80)
                print()

            # 3. 调用 LLM 2 (策略师)
            portfolio_data = interface_ec.evol.call_strategist(meta_prompt)
            portfolio = portfolio_data.get('portfolio', [])
            strategist_analysis = portfolio_data.get('analysis', 'N/A')
            print(f"[Strategist] Analysis: {strategist_analysis}")

            # 验证 portfolio 结构
            if not isinstance(portfolio, list) or len(portfolio) != self.r:
                print(
                    f"[ERROR] Strategist returned invalid portfolio (length {len(portfolio)}, expected {self.r}). Using fallback.")
                portfolio = [
                    {"strategy": "Fallback Exploration", "P_draft": self._get_fallback_explore_prompt()},
                    {"strategy": "Fallback Exploitation",
                     "P_draft": self._get_fallback_m2_prompt(population[0] if population else None)}
                ]
                last_feedback_to_strategist = f"Generation {pop}: Strategist failed to return a valid portfolio. Used fallback strategies."
            else:
                last_feedback_to_strategist = f"Generation {pop}: Strategist proposed {self.r} strategies. Analysis: {strategist_analysis}"

            # 4. (重构后) 直接使用 P_draft，无需 Engineer 审查
            final_p_drafts = []

            if not portfolio or len(portfolio) == 0:
                print("[WARNING] Strategist returned empty portfolio. Using fallback P_drafts.")
                final_p_drafts = [
                    {"P_draft": self._get_fallback_explore_prompt(), "desc": "Fallback Exploration"},
                    {"P_draft": self._get_fallback_m2_prompt(None), "desc": "Fallback Exploitation"}
                ]
            else:
                for i, item in enumerate(portfolio):
                    strategy_desc = item.get('strategy_description', item.get('strategy', f'Strategy {i + 1}'))
                    p_draft = item.get('P_draft', '')

                    if not p_draft:
                        print(f"  [WARNING] P_draft {i + 1} is empty, using fallback.")
                        p_draft = self._get_fallback_explore_prompt() if i % 2 == 0 else self._get_fallback_m2_prompt(
                            None)

                    final_p_drafts.append({"P_draft": p_draft, "desc": strategy_desc})
                    print(f"  Operator {i + 1}/{self.r}: {strategy_desc}")

            # 5. (重构后) 分配预算并并行执行生成
            num_valid_ops = len(final_p_drafts)
            base_budget_per_op = self.pop_size // num_valid_ops
            remaining_budget = self.pop_size % num_valid_ops
            print(f"[Execution] Distributing {self.pop_size} evaluations among {num_valid_ops} operator(s).")

            all_new_offsprings = []
            budgets = [base_budget_per_op] * num_valid_ops
            for i in range(remaining_budget):
                budgets[i] += 1

            for i, op_data in enumerate(final_p_drafts):
                p_draft = op_data.get('P_draft', '')
                op_desc = op_data.get('desc', f'Operator {i + 1}')
                op_budget = budgets[i]

                if not p_draft or op_budget == 0:
                    continue

                print(f"  Executing Operator {i + 1}/{num_valid_ops} ({op_desc[:40]}...) with budget {op_budget}...")
                if self.debug_mode:
                    print(f"    [DEBUG] P_draft (length: {len(p_draft)} chars):")
                    print(f"      {p_draft[:100]}{'...' if len(p_draft) > 100 else ''}")
                    print(f"    [DEBUG] Note: Worker will receive P_draft + parent samples + problem constraints\n")

                offsprings_from_op = interface_ec.generate_offspring_batch(
                    population,
                    selected_parents,  # 传递父代样本
                    p_draft,  # 传递P_draft
                    op_budget
                )
                all_new_offsprings.extend(offsprings_from_op)
                print(f"    Generated {len(offsprings_from_op)} valid offspring from operator {i + 1}.")

            # 6. 合并后代
            added_count = self.add2pop(population, all_new_offsprings)
            print(
                f"[Population] Added {added_count} new unique offspring to the pool (current pool size: {len(population)}).")

            # 7. 种群管理
            if not population:
                print("[ERROR] Population became empty before management. Cannot select next generation.")
                break

            population_before_manage = len(population)
            population = self.manage.population_management(population, self.pop_size)
            print(
                f"[Population] Selected top {len(population)} individuals from {population_before_manage} for the next generation.")

            # 8. 更新停滞计数器
            current_best_obj = population[0].get('objective') if population else float('inf')
            last_best_obj = float('inf')
            valid_history = [obj for obj in best_objective_history if isinstance(obj, (int, float, np.number))]
            if valid_history:
                last_best_obj = valid_history[-1]

            if isinstance(current_best_obj, (int, float, np.number)) and current_best_obj < last_best_obj:
                stagnation_count = 0
                best_objective_history.append(current_best_obj)
            else:
                stagnation_count += 1
                best_objective_history.append(last_best_obj)

            # 9. 保存结果
            try:
                gen_pop_filename = os.path.join(self.output_path, "results", "pops",
                                                f"population_generation_{pop + 1}.json")
                best_pop_filename = os.path.join(self.output_path, "results", "pops_best",
                                                 f"population_generation_{pop + 1}.json")
                with open(gen_pop_filename, 'w') as f:
                    json.dump(population, f, indent=4)
                if population:
                    with open(best_pop_filename, 'w') as f:
                        json.dump(population[0], f, indent=4)
            except Exception as e:
                print(f"Warning: Failed to save population files for generation {pop + 1}: {e}")

            # 保存本代的LLM交互记录
            try:
                interface_ec.evol.save_llm_interactions(self.output_path, generation=pop + 1)
            except Exception as e:
                print(f"Warning: Failed to save LLM interactions for generation {pop + 1}: {e}")

            elapsed_time = (time.time() - time_start) / 60
            best_obj_display = f"{current_best_obj:.6f}" if isinstance(current_best_obj,
                                                                       (int, float, np.number)) else "N/A"
            print(
                f"\n--- Gen {pop + 1}/{self.n_pop} completed. Best: {best_obj_display}, Stagnation: {stagnation_count}, Time: {elapsed_time:.1f} min ---")

        # === 进化结束后的总结 ===
        print("\n" + "=" * 70)
        print("SEC V6 EVOLUTION COMPLETED")
        print("=" * 70)
        print(f"Total generations run: {self.n_pop}")
        total_llm_calls = getattr(interface_ec.evol, 'llm_call_count', 'N/A')
        print(f"Total LLM calls (approx): {total_llm_calls}")
        print(f"Final population size: {len(population)}")
        if population:
            final_best_obj = population[0].get('objective', 'N/A')
            best_obj_display = f"{final_best_obj:.6f}" if isinstance(final_best_obj, (int, float, np.number)) else "N/A"
            print(f"Final best fitness: {best_obj_display}")
        else:
            print("Final population is empty.")
        print("=" * 70)

        # 保存完整的LLM交互历史
        try:
            interface_ec.evol.save_llm_interactions(self.output_path, generation=None)
            total_interactions = len(interface_ec.evol.llm_interaction_history)
            print(f"[LLM Log] Saved {total_interactions} total LLM interactions to llm_interactions/")
        except Exception as e:
            print(f"Warning: Failed to save complete LLM interaction history: {e}")


# 为了向后兼容，保留 EOH 类
EOH = EVOL
