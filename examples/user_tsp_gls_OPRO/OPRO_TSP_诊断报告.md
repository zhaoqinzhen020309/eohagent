# TSP OPRO vs 原始EoH 深度诊断报告

## 一、收敛效果对比

```
Generation    OPRO Gap%      原始EoH Gap%    性能差距
--------------------------------------------------------------
Gen 1        1.09833        0.96009         +0.13824 (14%)
Gen 3        0.48703        0.19338         +0.29365 (152%)
Gen 5        0.48703        0.01843         +0.46860 (2543%!)
Gen 10       0.03205        0.01843         +0.01362 (74%)
Gen 20       0.02886        0.01578         +0.01308 (83%)
```

**结论**：OPRO最终性能是原始EoH的1.83倍（差83%），虽然绝对值都很好（<0.03%），但存在明显差距。

---

## 二、关键问题诊断

### 【问题1】P_draft缺乏多样性和创新性

#### 观察到的P_draft模式：

**Exploitation策略（20代中的主要模式）**：
- Gen 1, 5, 15, 20: "Combine the most effective penalty components/mechanisms from all parents"
- Gen 10: "Identify the simplest effective penalty mechanism from all parents"
- **问题**：
  1. 20代中几乎全是"combine"的变体，缺乏其他动词（identify, extract, isolate, merge等）
  2. "most effective"违反了AVOID规则的精神（类似"top-performing"）
  3. 过于通用，Worker LLM理解为"把所有复杂组件堆在一起"

**Exploration策略（20代中的主要模式）**：
- Gen 1, 5, 10, 15, 20: "Explore/Create a fundamentally different mathematical penalty approach/formulation"
- **问题**：
  1. 20代中几乎完全相同，只是"explore"和"create"、"approach"和"formulation"互换
  2. "fundamentally different"虽然开放，但太宽泛，Worker不知道往哪个方向不同
  3. 没有提供任何具体的探索方向提示

#### Strategist进入"舒适区"：
- Strategist LLM找到了一套"安全"的模板后，就不再创新了
- 这些P_draft在语法和长度上符合要求，但**缺乏演化多样性**

---

### 【问题2】代码演化路径的根本差异

#### 原始EoH的创新路径（前5代）：

**Gen 1 (Gap 0.96%)**: 
```python
# 简单的线性惩罚
penalty_factor = 1.2
usage_penalty = 1 + usage_weight * edge_n_used[from_node, to_node]
```

**Gen 3 (Gap 0.19%)** ← **突破性创新！**
```python
# 发明了"逆向惩罚"机制
penalty_factor = base_penalty * (1.0 + (max_usage - usage_freq) / (max_usage - min_usage + 1e-8))
# 使用次数少的边 → 惩罚更大（反直觉但有效！）
```

**Gen 5 (Gap 0.01843%)** ← **质的飞跃！**
```python
# 引入全局统计 + 归一化 + tanh非线性
normalized_usage = edge_n_used[u, v] / (max_usage + 1e-8)
usage_variability = np.std(edge_n_used) / (mean_usage + 1e-8)
penalty = 1.0 + dynamic_factor * (1.5 + 0.5 * np.tanh(5.0 * (normalized_usage - 0.3)))
```

**关键观察**：
- 原始EoH在Gen 3找到了**逆向惩罚**这个非常规策略
- Gen 5进一步引入**全局统计**（mean, std）和**归一化**
- 这些都是**不明显的、需要创造性思维的创新**

---

#### OPRO的渐进式路径（前5代）：

**Gen 1 (Gap 1.09%)**:
```python
# 标准GLS惩罚
penalty_factor = 1.2
usage_penalty = 1 + usage_weight * edge_n_used[from_node, to_node]
```

**Gen 3 (Gap 0.48%)**:
```python
# 添加了log和exp，但仍是标准正向惩罚
dynamic_penalty_factor = base_penalty_factor + np.log1p(edge_n_used[from_node, to_node]) * 0.05
exponential_decay = np.exp(-decay_rate * edge_n_used[from_node, to_node])
```

**Gen 5 (Gap 0.48%)** ← **停滞！**
```python
# 与Gen 3完全相同的代码
# 说明演化停滞了
```

**Gen 10 (Gap 0.03%)** ← **突破，但靠的是"堆叠"**
```python
# 添加了更多组件：inverse_scaling + gaussian_penalty + adaptive_annealing + jitter
inverse_usage_scaling = max_usage / (edge_n_used[from_node, to_node] + 1)
gaussian_penalty = np.exp(-usage_weight * edge_n_used[from_node, to_node])
adaptive_annealing = cooling_rate ** edge_n_used[from_node, to_node]
probabilistic_jitter = np.random.normal(1, sigma)
# 所有组件相乘
```

**关键观察**：
- OPRO从未发现"逆向惩罚"或"全局统计"这类创新策略
- OPRO的改进来自**堆叠更多标准组件**（log + exp + sigmoid + cooling + jitter）
- 这是一种**渐进式优化**，而非**创新式突破**

---

### 【问题3】为什么OPRO缺乏创新？

#### 根本原因分析：

**1. P_draft过于通用和重复**
- "Combine the most effective penalty components" → Worker理解为"把所有能想到的技术都加上"
- "Explore a fundamentally different approach" → Worker不知道"different"的具体方向，倾向于使用熟悉的数学函数（log, exp, sigmoid）

**2. Worker LLM的"安全偏好"**
- LLM训练数据中常见的penalty机制：线性、指数、对数、sigmoid
- LLM很少见到"逆向惩罚"（因为这在文献中较少，且反直觉）
- 当P_draft说"create a new approach"时，LLM倾向于组合熟悉的组件，而非发明新机制

**3. 缺乏"负面引导"**
- 我们的P_draft只说"做什么"（combine, explore），没有说"不要做什么"或"避免什么"
- 例如，没有P_draft说："avoid using only forward penalties, try inverse relationships"
- 没有P_draft说："don't just stack components, try a completely different penalty philosophy"

---

## 三、与Bin Packing OPRO的成功对比

### 为什么Bin Packing OPRO成功？

回顾Bin Packing问题的特点：
1. **问题空间更明确**：scoring function有清晰的目标（capacity utilization, item count, etc.）
2. **创新路径更直接**：结合不同的scoring维度（bins_remaining, item_size, capacity_left）就能产生好效果
3. **Worker LLM更熟悉**：bin packing heuristics在文献中更常见，LLM见过更多类似的代码

### TSP-GLS的特殊性：

1. **问题空间更抽象**：distance matrix update的"好"策略不直观
2. **需要反直觉创新**：逆向惩罚、全局统计这类策略在文献中较少
3. **Worker LLM经验不足**：TSP的GLS penalty机制在训练数据中可能较少

---

## 四、改进建议

### 【建议1】增强P_draft的多样性和方向性

当前Strategist Prompt的问题：
- 示例P_draft过于通用："combine different techniques", "explore fundamentally different"
- 没有给出具体的探索维度

**改进方向**：

在Strategist Prompt中添加"探索维度列表"：

```
**Potential Exploration Dimensions for TSP-GLS** (for inspiration, not prescriptive):

- Penalty direction: forward (high usage → high penalty) vs. inverse (low usage → high penalty)
- Scaling approach: linear vs. exponential vs. power-law vs. logarithmic
- Scope: local (per-edge) vs. global (considering all edges' statistics)
- Adaptation: static vs. dynamic (usage-dependent decay)
- Stochasticity: deterministic vs. probabilistic
- Normalization: absolute values vs. relative to mean/max/min

Your P_draft should hint at ONE of these dimensions (or others), without naming specific methods.
```

这样Strategist可以生成更具体的P_draft：
- "explore inverse relationships between usage and penalty"
- "incorporate global edge statistics into local penalty"
- "try a probabilistic penalty mechanism"

### 【建议2】添加"负面引导"机制

在Strategist Prompt中添加：

```
**What to AVOID suggesting** (if the population shows these patterns):

- If all parents use forward penalties (high usage → high penalty): 
  Consider suggesting: "try inverse penalty relationships"
  
- If all parents only use local edge information:
  Consider suggesting: "incorporate global usage patterns"
  
- If all parents are deterministic:
  Consider suggesting: "add controlled randomness"
```

### 【建议3】使用"对比式"P_draft

当前P_draft格式：
```json
"P_draft": "Combine the most effective penalty components from all parents"
```

改进为**对比式**：
```json
"P_draft": "Unlike parents' forward penalties, try inverse usage relationships"
"P_draft": "Move beyond per-edge penalties to global statistical awareness"
"P_draft": "Replace deterministic factors with controlled stochastic elements"
```

这种格式明确告诉Worker：
1. 父代在做什么（"forward penalties", "per-edge"）
2. 你应该做什么不同的事（"inverse", "global"）

---

## 五、是否需要修改Prompt？

### 短期（当前实验）：

**不建议修改**，原因：
1. OPRO已经取得了不错的结果（0.02886%，虽然不如原始EoH，但绝对性能很好）
2. 说明当前框架是**可行的**，只是**不够优化**
3. 我们需要先在FSSP问题上验证当前框架的通用性

### 中期（FSSP实验后）：

**建议修改**，如果FSSP也出现类似问题：
1. 实施"探索维度列表"
2. 添加"负面引导"机制
3. 考虑"对比式"P_draft格式

### 长期（研究方向）：

**更激进的改进**：
1. **动态Prompt调整**：根据种群特征（是否同质化）动态调整Strategist Prompt
2. **分层P_draft**：Strategist不仅生成方向，还生成"禁止区域"
3. **Worker反馈**：Worker告诉Strategist哪些P_draft理解不清楚

---

## 六、总结

### OPRO TSP实验的核心发现：

1. **框架有效性**：OPRO框架是可行的，能在TSP-GLS问题上收敛到很好的解（0.02886%）

2. **性能差距**：比原始EoH差83%（0.02886% vs 0.01578%）

3. **根本原因**：
   - P_draft缺乏多样性（20代中几乎相同的模板）
   - Worker LLM倾向于"渐进式堆叠"而非"创新式突破"
   - 没有发现"逆向惩罚"、"全局统计"等关键创新

4. **关键洞察**：
   - 原始EoH的固定算子虽然"僵硬"，但能通过**组合和突变**产生**意外的创新**
   - OPRO的灵活P_draft虽然"开放"，但LLM倾向于**安全的渐进优化**
   - **这是创造力vs稳定性的权衡**

5. **下一步**：
   - 先在FSSP问题上验证当前框架
   - 如果FSSP也出现类似问题，再系统性改进Strategist Prompt
   - 保持当前框架的简洁性，不要过早优化

---

## 七、API调用效率对比

```
原始EoH (20代):
  - 固定5个算子 × 20代 = 约100次算子调用
  - 每次算子调用可能需要1-2次LLM调用（Engineer + Code generation）
  - 总计：约150-200次LLM调用
  
OPRO (20代):
  - 每代：1次Strategist调用 + 10次Worker调用 = 11次LLM调用
  - 总计：20 × 11 = 220次LLM调用
  
效率对比：
  - OPRO略多（220 vs 150-200），但在同一数量级
  - OPRO的优势：只用2个算子而非5个，更简洁
  - 但性能差距说明：简洁不等于高效
```

**注**：用户之前提到"原始eoh是2000多次的API调用"可能指的是完整的multi-run实验（多次独立运行），而非单次20代的调用。

---

## 八、最终评价

**OPRO TSP实验：B+评级**

优点：
- ✓ 框架成功运行，双LLM协同工作正常
- ✓ 最终性能良好（0.02886%，实用价值高）
- ✓ P_draft基本符合设计规范（无明显违规）
- ✓ 证明了OPRO框架的可迁移性（从Bin Packing到TSP）

不足：
- ✗ 性能不如原始EoH（差83%）
- ✗ P_draft缺乏多样性和创新性
- ✗ Worker LLM偏向渐进式优化而非创新式突破

**建议**：保留当前框架，先验证FSSP问题，再考虑系统性改进。

