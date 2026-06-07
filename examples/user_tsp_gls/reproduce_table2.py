# reproduce_table2_complete.py
import sys
import json
import numpy as np
import types
import time

sys.path.append('utils')
sys.path.append('gls')

# TSPLib最优解
optimal_solutions = {
    'rd100.tsp': 7910.396,
    'pr124.tsp': 59030.0,
    'bier127.tsp': 118282.0,
    'kroA150.tsp': 26524.0,
    'u159.tsp': 42080.0,
    'kroB200.tsp': 29437.0
}


def main():
    print("=== 一步到位复现Table 2 ===")

    # 1. 直接从训练结果提取最佳算法
    print("提取最佳算法...")
    with open('./results/pops/population_generation_20.json', 'r') as f:
        final_pop = json.load(f)

    best_individual = final_pop[0]
    print(f"最佳个体fitness: {best_individual['objective']}")

    # 创建算法模块
    algorithm_module = types.ModuleType("algorithm_module")
    exec(best_individual['code'], algorithm_module.__dict__)

    # 2. 导入必要的模块
    try:
        from readTSPLib import read_instance
        print("✓ TSPLib读取模块导入成功")
    except Exception as e:
        print(f"✗ TSPLib读取失败: {e}")
        return

    try:
        from gls_run import solve_instance
        print("✓ GLS求解模块导入成功")
    except Exception as e:
        print(f"✗ GLS模块导入失败: {e}")
        print("尝试手动修复utils导入...")
        # 创建空的__init__.py如果不存在
        try:
            with open('utils/__init__.py', 'w') as f:
                f.write("from .utils import *\n")
            from gls_run import solve_instance
            print("✓ 修复后GLS模块导入成功")
        except Exception as e2:
            print(f"✗ 修复失败: {e2}")
            return

    # 3. 格式转换函数
    def networkx_to_matrix_coords(G, scale_factor):
        nodes = sorted(G.nodes())
        n = len(nodes)

        # 距离矩阵
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_matrix[i][j] = G[nodes[i]][nodes[j]]['weight'] * scale_factor

        # 虚拟坐标
        coords = np.random.rand(n, 2) * 100
        return distance_matrix, coords

    # 4. 测试所有TSPLib实例
    instances = ['rd100.tsp', 'pr124.tsp', 'bier127.tsp',
                 'kroA150.tsp', 'u159.tsp', 'kroB200.tsp']

    results = {}

    for instance in instances:
        print(f"\n=== 测试 {instance} ===")

        try:
            # 读取TSPLib数据
            G, scale_factor = read_instance(f'./TestingData/TSPLib/{instance}')
            distance_matrix, coords = networkx_to_matrix_coords(G, scale_factor)
            opt_cost = optimal_solutions[instance]

            print(f"城市数: {len(G.nodes())}, 最优解: {opt_cost}, 缩放: {scale_factor}")

            # 使用现有的solve_instance函数
            gap = solve_instance(0, opt_cost, distance_matrix, coords, 60, 1000, 1, algorithm_module)

            print(f"Gap: {gap:.2f}%")
            results[instance] = gap

        except Exception as e:
            print(f"测试失败: {e}")
            import traceback
            traceback.print_exc()
            results[instance] = None

    # 5. 输出Table 2格式结果
    print("\n" + "=" * 40)
    print("Table 2 复现结果")
    print("=" * 40)
    print(f"{'Instance':<12} {'Gap (%)':<10}")
    print("-" * 25)

    for instance, gap in results.items():
        name = instance.replace('.tsp', '')
        gap_str = f"{gap:.2f}" if gap is not None else "失败"
        print(f"{name:<12} {gap_str:<10}")

    # 6. 与论文结果对比
    paper_results = {
        'rd100': 0.01, 'pr124': 0.00, 'bier127': 0.42,
        'kroA150': 0.00, 'u159': 0.00, 'kroB200': 0.20
    }

    print(f"\n{'Instance':<12} {'论文结果':<10} {'复现结果':<10} {'差异':<10}")
    print("-" * 45)
    for instance, gap in results.items():
        name = instance.replace('.tsp', '')
        paper_gap = paper_results.get(name, 'N/A')
        our_gap = f"{gap:.2f}" if gap is not None else "失败"
        diff = f"{gap - paper_gap:.2f}" if gap is not None and isinstance(paper_gap, float) else "N/A"
        print(f"{name:<12} {paper_gap:<10} {our_gap:<10} {diff:<10}")


if __name__ == "__main__":
    main()