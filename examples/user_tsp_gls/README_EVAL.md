# TSP-GLS 评估指南

## 📋 评估流程

### 1. 提取最佳启发式算法

从进化结果中提取最佳个体的代码：

```bash
# 提取第20代的最佳个体
python extract_best_tsp.py ./results/pops_best/population_generation_20.json --output best_heuristic.py
```

**输出**：
- `best_heuristic.py`：包含 `update_edge_distance` 函数的Python文件

### 2. 在TSPLib测试集上评估

运行评估脚本：

```bash
python eval_tsp.py --heuristic best_heuristic.py
```

**测试集**：
- rd100 (100个城市)
- pr124 (124个城市)
- bier127 (127个城市)
- kroA150 (150个城市)
- u159 (159个城市)
- kroB200 (200个城市)

**评估指标**：
- Gap %：`(found_cost - optimal_cost) / optimal_cost * 100`
- 越低越好，0%表示找到最优解

---

## 🔄 完整示例

```bash
# 1. 提取最佳个体
python extract_best_tsp.py ./results/pops_best/population_generation_20.json --output best_heuristic.py

# 输出示例：
# ============================================================
# 提取TSP-GLS最佳启发式算法
# ============================================================
# 读取文件: ./results/pops_best/population_generation_20.json
# 最佳个体fitness: 0.01578
# 算法描述: An algorithm that applies an exponential penalty...
# 
# 最佳个体代码:
# --------------------------------------------------
# import numpy as np
# def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
#     ...
# --------------------------------------------------
# 已保存到: best_heuristic.py

# 2. 评估
python eval_tsp.py

# 输出示例：
# ============================================================
# TSPLib测试集评估结果
# ============================================================
# Instance        Cities     Gap (%)   
# ------------------------------------------------------------
# rd100           100        0.0100    
# pr124           124        0.0000    
# bier127         127        0.4200    
# kroA150         150        0.0000    
# u159            159        0.0000    
# kroB200         200        0.2000    
# ------------------------------------------------------------
# 平均Gap                     0.1050
# ============================================================
```

---

## 📊 与Bin Packing的对比

| 项目 | Bin Packing | TSP-GLS |
|------|-------------|---------|
| **提取脚本** | `extract_best_heuristic.py` | `extract_best_tsp.py` |
| **评估脚本** | `runEval.py` | `eval_tsp.py` |
| **函数名** | `score(item, bins)` | `update_edge_distance(...)` |
| **训练集** | Weibull 5k (64实例) | TSPAEL64.pkl (64实例) |
| **测试集** | 9个Weibull测试集 | 6个TSPLib实例 |
| **评估指标** | Excess % | Gap % |
| **目标** | 最小化bins数量 | 最小化路径长度 |

---

## 🛠️ 故障排除

### 问题1：找不到模块

```
错误: TSPLib读取失败: No module named 'readTSPLib'
```

**解决方案**：
```bash
# 确保在 user_tsp_gls 目录下运行
cd examples/user_tsp_gls
python eval_tsp.py
```

### 问题2：文件不存在

```
错误: 文件 best_heuristic.py 不存在
```

**解决方案**：
```bash
# 先提取最佳个体
python extract_best_tsp.py ./results/pops_best/population_generation_20.json
```

### 问题3：评估超时

如果某个实例评估时间过长，可以修改 `eval_tsp.py` 中的参数：

```python
gap = solve_instance(
    ...
    time_limit=60,      # 增加时间限制（秒）
    ite_max=1000,       # 减少最大迭代次数
    ...
)
```

---

## 📝 注意事项

1. **训练vs测试**：
   - 训练fitness（如0.01578）是在64个随机实例上的平均Gap
   - 测试结果是在6个TSPLib标准实例上的Gap
   - 两者不直接可比，但都反映算法性能

2. **随机性**：
   - GLS算法有随机性（初始解、扰动等）
   - 每次运行结果可能略有不同
   - 论文中的结果是多次运行的平均值

3. **计算时间**：
   - 每个实例评估约需60秒（time_limit）
   - 6个实例总计约6分钟
   - 可以根据需要调整time_limit

---

## 🎯 论文Table 2对比

论文中报告的结果（EoH方法）：

| Instance | Gap (%) |
|----------|---------|
| rd100    | 0.01    |
| pr124    | 0.00    |
| bier127  | 0.42    |
| kroA150  | 0.00    |
| u159     | 0.00    |
| kroB200  | 0.20    |
| **平均** | **0.11** |

你可以将你的结果与此对比！

