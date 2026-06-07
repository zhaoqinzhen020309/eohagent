import json
import os
import numpy as np

# 读取所有代的fitness数据
results_dir = r"C:\Users\48520\Desktop\eoh多目标\examples\user_bp_online_OPRO\results"
pops_best_dir = os.path.join(results_dir, "pops_best")
llm_dir = os.path.join(results_dir, "llm_interactions")

# 收集所有代的fitness
fitness_data = []
for gen in range(1, 21):
    filepath = os.path.join(pops_best_dir, f"population_generation_{gen}.json")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        fitness_data.append({
            'generation': gen,
            'fitness': data['objective'],
            'code': data['code']
        })

# 打印fitness轨迹
print("=" * 80)
print("FITNESS TRAJECTORY (20 Generations)")
print("=" * 80)
for item in fitness_data:
    print(f"Gen {item['generation']:2d}: {item['fitness']:.5f}")

# 计算改进率
print("\n" + "=" * 80)
print("IMPROVEMENT ANALYSIS")
print("=" * 80)
for i in range(1, len(fitness_data)):
    prev_fit = fitness_data[i-1]['fitness']
    curr_fit = fitness_data[i]['fitness']
    improvement = prev_fit - curr_fit
    improvement_pct = (improvement / prev_fit) * 100 if prev_fit > 0 else 0

    if improvement > 0:
        print(f"Gen {i} -> {i+1}: {prev_fit:.5f} -> {curr_fit:.5f} (↓{improvement:.5f}, {improvement_pct:.2f}% improvement)")
    elif improvement < 0:
        print(f"Gen {i} -> {i+1}: {prev_fit:.5f} -> {curr_fit:.5f} (↑{abs(improvement):.5f}, regression)")
    else:
        print(f"Gen {i} -> {i+1}: {prev_fit:.5f} -> {curr_fit:.5f} (no change)")

# 识别关键节点
print("\n" + "=" * 80)
print("KEY EVOLUTIONARY NODES IDENTIFICATION")
print("=" * 80)

# 1. 初始代
print("\n[NODE 1] Generation 1: Starting Point")
print(f"  Fitness: {fitness_data[0]['fitness']:.5f}")

# 2. 首次显著提升 (Gen 3->4)
print("\n[NODE 2] Generation 4: First Major Breakthrough")
gen3_fit = fitness_data[2]['fitness']
gen4_fit = fitness_data[3]['fitness']
improvement = (gen3_fit - gen4_fit) / gen3_fit * 100
print(f"  Fitness: {gen4_fit:.5f}")
print(f"  Improvement from Gen 3: {improvement:.2f}%")
print(f"  First significant jump in performance")

# 3. 第二次显著提升 (Gen 4->6)
print("\n[NODE 3] Generation 6: Second Major Improvement")
gen6_fit = fitness_data[5]['fitness']
improvement = (gen4_fit - gen6_fit) / gen4_fit * 100
print(f"  Fitness: {gen6_fit:.5f}")
print(f"  Improvement from Gen 4: {improvement:.2f}%")
print(f"  Introduced power-law penalty terms")

# 4. 持续优化 (Gen 8)
print("\n[NODE 4] Generation 8: Refinement Phase")
gen8_fit = fitness_data[7]['fitness']
improvement = (gen6_fit - gen8_fit) / gen6_fit * 100
print(f"  Fitness: {gen8_fit:.5f}")
print(f"  Improvement from Gen 6: {improvement:.2f}%")
print(f"  Fine-tuned penalty structure")

# 5. 接近收敛 (Gen 11)
print("\n[NODE 5] Generation 11: Near Convergence")
gen11_fit = fitness_data[10]['fitness']
improvement = (gen8_fit - gen11_fit) / gen8_fit * 100
print(f"  Fitness: {gen11_fit:.5f}")
print(f"  Improvement from Gen 8: {improvement:.2f}%")
print(f"  Algorithm stabilization")

# 6. 最终收敛 (Gen 11->20保持不变)
print("\n[NODE 6] Generation 11-20: Convergence")
print(f"  Fitness: {fitness_data[10]['fitness']:.5f} (maintained through Gen 20)")
print(f"  Evolution has converged to local optimum")

# 总体统计
print("\n" + "=" * 80)
print("OVERALL STATISTICS")
print("=" * 80)
initial_fitness = fitness_data[0]['fitness']
final_fitness = fitness_data[-1]['fitness']
total_improvement = (initial_fitness - final_fitness) / initial_fitness * 100
print(f"Initial Fitness (Gen 1):  {initial_fitness:.5f}")
print(f"Final Fitness (Gen 20):   {final_fitness:.5f}")
print(f"Total Improvement:        {total_improvement:.2f}%")
print(f"Number of unique fitness: {len(set([d['fitness'] for d in fitness_data]))}")

# 分析策略使用情况
print("\n" + "=" * 80)
print("STRATEGY ANALYSIS")
print("=" * 80)

key_gens = [1, 3, 4, 6, 8, 11]
for gen in key_gens:
    llm_file = os.path.join(llm_dir, f"llm_interactions_gen_{gen}.json")
    with open(llm_file, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)
        if llm_data and 'parsed_result' in llm_data[0]:
            portfolio = llm_data[0]['parsed_result']['portfolio']
            print(f"\nGeneration {gen}:")
            for strategy in portfolio:
                print(f"  - {strategy['strategy_type']}: \"{strategy['P_draft']}\"")

print("\n" + "=" * 80)
print("CODE EVOLUTION ANALYSIS")
print("=" * 80)

# 分析关键代的代码特征
key_code_gens = [1, 4, 6, 8, 11]
for gen_idx in key_code_gens:
    code = fitness_data[gen_idx - 1]['code']
    fitness = fitness_data[gen_idx - 1]['fitness']

    print(f"\n{'='*60}")
    print(f"Generation {gen_idx} (Fitness: {fitness:.5f})")
    print('='*60)

    # 分析代码特征
    features = []
    if 'optimal_fullness' in code:
        features.append("Uses optimal_fullness metric")
    if '**' in code and '-1.2' in code:
        features.append("Power-law size penalty (x^-1.2)")
    if 'np.log1p' in code:
        features.append("Logarithmic penalty")
    if 'diversity' in code:
        features.append("Diversity consideration")
    if 'proximity_optimal' in code:
        features.append("Proximity to optimal")
    if 'item_count' in code or 'num_items' in code:
        features.append("Item count penalty")
    if '** 0.7' in code or '** 0.5' in code or '** 0.4' in code:
        features.append("Non-linear transformations")

    print("Key Features:")
    for feature in features:
        print(f"  - {feature}")

    # 提取权重系数
    if gen_idx == 1:
        print("\nWeights: Simple linear combination")
    elif gen_idx >= 4:
        print("\nWeight Structure:")
        lines = code.split('\n')
        for line in lines:
            if 'scores[valid_bins]' in line and '=' in line:
                print(f"  {line.strip()}")
                break

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)