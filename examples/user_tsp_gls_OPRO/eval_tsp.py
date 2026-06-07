#!/usr/bin/env python3
"""
评估TSP-GLS启发式算法在测试集上的性能
用法: python eval_tsp.py [--heuristic <heuristic_file>]
"""

import sys
import json
import numpy as np
import types
import time
import argparse
import os

# 添加模块搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# TSPLib最优解 (TABLE 11 - All 29 instances)
OPTIMAL_SOLUTIONS = {
    'eil51.tsp': 426,
    'berlin52.tsp': 7542,
    'st70.tsp': 675,
    'eil76.tsp': 538,
    'pr76.tsp': 108159,
    'rat99.tsp': 1211,
    'rd100.tsp': 7910.396,
    'kroA100.tsp': 21282,
    'kroB100.tsp': 22141,
    'kroC100.tsp': 20749,
    'kroD100.tsp': 21294,
    'kroE100.tsp': 22068,
    'eil101.tsp': 629,
    'lin105.tsp': 14379,
    'pr107.tsp': 44303,
    'pr124.tsp': 59030.0,
    'bier127.tsp': 118282.0,
    'ch130.tsp': 6110,
    'pr136.tsp': 96772,
    'pr144.tsp': 58537,
    'ch150.tsp': 6528,
    'kroA150.tsp': 26524.0,
    'kroB150.tsp': 26130,
    'pr152.tsp': 73682,
    'u159.tsp': 42080.0,
    'rat195.tsp': 2323,
    'd198.tsp': 15780,
    'kroA200.tsp': 29368,
    'kroB200.tsp': 29437.0
}


def load_heuristic(heuristic_path='best_heuristic.py'):
    """加载启发式算法模块"""
    print(f"Loading heuristic: {heuristic_path}")

    try:
        with open(heuristic_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except FileNotFoundError:
        print(f"Error: File {heuristic_path} not found")
        print("Please run first: python extract_best_tsp.py <population_json>")
        sys.exit(1)

    # 创建算法模块
    algorithm_module = types.ModuleType("algorithm_module")
    try:
        exec(code, algorithm_module.__dict__)
    except Exception as e:
        print(f"Error: Code execution failed - {e}")
        sys.exit(1)

    # 验证函数存在
    if not hasattr(algorithm_module, 'update_edge_distance'):
        print("Error: Function 'update_edge_distance' not found in code")
        sys.exit(1)

    print("[OK] Heuristic loaded successfully\n")
    return algorithm_module


def networkx_to_matrix_coords(G, scale_factor):
    """将NetworkX图转换为距离矩阵和坐标"""
    nodes = sorted(G.nodes())
    n = len(nodes)

    # 距离矩阵
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = G[nodes[i]][nodes[j]]['weight'] * scale_factor

    # 虚拟坐标（用于GLS算法）- 设置固定随机种子以确保可重复性
    np.random.seed(2024)
    coords = np.random.rand(n, 2) * 100
    return distance_matrix, coords


def evaluate_on_tsplib(algorithm_module):
    """在TSPLib测试集上评估算法"""

    # 导入必要模块
    try:
        from utils import readTSPLib
        read_instance = readTSPLib.read_instance
        print("[OK] TSPLib reading module imported")
    except Exception as e:
        print(f"[ERROR] TSPLib reading failed: {e}")
        print(f"[DEBUG] Current sys.path: {sys.path[:3]}")
        return {}

    try:
        from gls import gls_run
        solve_instance = gls_run.solve_instance
        print("[OK] GLS solver module imported\n")
    except Exception as e:
        print(f"[ERROR] GLS module import failed: {e}")
        print(f"[DEBUG] Current sys.path: {sys.path[:3]}")
        return {}

    # 测试实例列表 (TABLE 11 - All 29 instances)
    instances = [
        'eil51.tsp', 'berlin52.tsp', 'st70.tsp', 'eil76.tsp', 'pr76.tsp',
        'rat99.tsp', 'rd100.tsp', 'kroA100.tsp', 'kroB100.tsp', 'kroC100.tsp',
        'kroD100.tsp', 'kroE100.tsp', 'eil101.tsp', 'lin105.tsp', 'pr107.tsp',
        'pr124.tsp', 'bier127.tsp', 'ch130.tsp', 'pr136.tsp', 'pr144.tsp',
        'ch150.tsp', 'kroA150.tsp', 'kroB150.tsp', 'pr152.tsp', 'u159.tsp',
        'rat195.tsp', 'd198.tsp', 'kroA200.tsp', 'kroB200.tsp'
    ]

    results = {}

    print("=" * 60)
    print("Start Evaluation on TSPLib Test Set")
    print("=" * 60)

    for instance in instances:
        print(f"\nTesting {instance}...")

        try:
            # 读取TSPLib数据
            G, scale_factor = read_instance(f'./TestingData/TSPLib/{instance}')
            distance_matrix, coords = networkx_to_matrix_coords(G, scale_factor)
            opt_cost = OPTIMAL_SOLUTIONS[instance]

            n_cities = len(G.nodes())
            print(f"  Cities: {n_cities}, Optimal: {opt_cost:.2f}, Scale: {scale_factor}")

            # 运行GLS求解
            start_time = time.time()
            gap = solve_instance(
                0,  # instance index
                opt_cost,  # optimal cost
                distance_matrix,  # distance matrix
                coords,  # coordinates
                60,  # time_limit: 60秒时间限制
                1000,  # ite_max: 最多1000次迭代
                1,  # perturbation_moves
                algorithm_module  # heuristic
            )
            elapsed = time.time() - start_time

            print(f"  Gap: {gap:.4f}% (Time: {elapsed:.1f}s)")
            results[instance] = gap

        except Exception as e:
            print(f"  Test failed: {e}")
            import traceback
            traceback.print_exc()
            results[instance] = None

    return results


def print_results(results, output_file='results.txt'):
    """打印评估结果并保存到文件"""

    city_counts = {
        'eil51.tsp': 51, 'berlin52.tsp': 52, 'st70.tsp': 70, 'eil76.tsp': 76,
        'pr76.tsp': 76, 'rat99.tsp': 99, 'rd100.tsp': 100, 'kroA100.tsp': 100,
        'kroB100.tsp': 100, 'kroC100.tsp': 100, 'kroD100.tsp': 100, 'kroE100.tsp': 100,
        'eil101.tsp': 101, 'lin105.tsp': 105, 'pr107.tsp': 107, 'pr124.tsp': 124,
        'bier127.tsp': 127, 'ch130.tsp': 130, 'pr136.tsp': 136, 'pr144.tsp': 144,
        'ch150.tsp': 150, 'kroA150.tsp': 150, 'kroB150.tsp': 150, 'pr152.tsp': 152,
        'u159.tsp': 159, 'rat195.tsp': 195, 'd198.tsp': 198, 'kroA200.tsp': 200,
        'kroB200.tsp': 200
    }

    # 准备输出内容
    lines = []
    lines.append("=" * 60)
    lines.append("TSPLib Test Set Evaluation Results")
    lines.append("=" * 60)
    lines.append(f"{'Instance':<15} {'Cities':<10} {'Gap (%)':<10}")
    lines.append("-" * 60)

    valid_gaps = []
    for instance, gap in results.items():
        name = instance.replace('.tsp', '')
        cities = city_counts.get(instance, 'N/A')
        if gap is not None:
            gap_str = f"{gap:.4f}"
            valid_gaps.append(gap)
        else:
            gap_str = "FAILED"
        lines.append(f"{name:<15} {cities:<10} {gap_str:<10}")

    if valid_gaps:
        avg_gap = np.mean(valid_gaps)
        lines.append("-" * 60)
        lines.append(f"{'Average Gap':<15} {'':<10} {avg_gap:.4f}")

    lines.append("=" * 60)

    # 打印到控制台
    print("\n" + "\n".join(lines))

    # 保存到文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"\nWarning: Failed to save results to file: {e}")


def main():
    parser = argparse.ArgumentParser(description='评估TSP-GLS启发式算法')
    parser.add_argument('--heuristic', default='best_heuristic.py',
                        help='启发式算法文件路径 (默认: best_heuristic.py)')

    args = parser.parse_args()

    # 加载启发式算法
    algorithm_module = load_heuristic(args.heuristic)

    # 在TSPLib上评估
    results = evaluate_on_tsplib(algorithm_module)

    # 打印结果
    if results:
        print_results(results)
    else:
        print("\nEvaluation failed, no results obtained")
        sys.exit(1)


if __name__ == '__main__':
    main()

