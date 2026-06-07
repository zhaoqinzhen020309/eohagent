# reproduce_table2_complete.py
import sys
import json
import numpy as np
import types
import time
import os

# 添加模块搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# TSPLib最优解 (TABLE 11 - All 29 instances)
optimal_solutions = {
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
        from utils import readTSPLib
        read_instance = readTSPLib.read_instance
        print("✓ TSPLib读取模块导入成功")
    except Exception as e:
        print(f"✗ TSPLib读取失败: {e}")
        print(f"[DEBUG] Current sys.path: {sys.path[:3]}")
        return

    try:
        from gls import gls_run
        solve_instance = gls_run.solve_instance
        print("✓ GLS求解模块导入成功")
    except Exception as e:
        print(f"✗ GLS模块导入失败: {e}")
        print(f"[DEBUG] Current sys.path: {sys.path[:3]}")
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

    # 4. 测试所有TSPLib实例 (TABLE 11 - All 29 instances)
    instances = [
        'eil51.tsp', 'berlin52.tsp', 'st70.tsp', 'eil76.tsp', 'pr76.tsp',
        'rat99.tsp', 'rd100.tsp', 'kroA100.tsp', 'kroB100.tsp', 'kroC100.tsp',
        'kroD100.tsp', 'kroE100.tsp', 'eil101.tsp', 'lin105.tsp', 'pr107.tsp',
        'pr124.tsp', 'bier127.tsp', 'ch130.tsp', 'pr136.tsp', 'pr144.tsp',
        'ch150.tsp', 'kroA150.tsp', 'kroB150.tsp', 'pr152.tsp', 'u159.tsp',
        'rat195.tsp', 'd198.tsp', 'kroA200.tsp', 'kroB200.tsp'
    ]

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

    # 5. 输出Table 2格式结果并保存到文件
    output_file = 'results.txt'

    # 准备输出内容
    lines = []
    lines.append("=" * 40)
    lines.append("Table 2 复现结果")
    lines.append("=" * 40)
    lines.append(f"{'Instance':<12} {'Gap (%)':<10}")
    lines.append("-" * 25)

    for instance, gap in results.items():
        name = instance.replace('.tsp', '')
        gap_str = f"{gap:.2f}" if gap is not None else "失败"
        lines.append(f"{name:<12} {gap_str:<10}")

    # 6. 与论文结果对比
    paper_results = {
        'rd100': 0.01, 'pr124': 0.00, 'bier127': 0.42,
        'kroA150': 0.00, 'u159': 0.00, 'kroB200': 0.20
    }

    lines.append("")
    lines.append(f"{'Instance':<12} {'论文结果':<10} {'复现结果':<10} {'差异':<10}")
    lines.append("-" * 45)
    for instance, gap in results.items():
        name = instance.replace('.tsp', '')
        paper_gap = paper_results.get(name, 'N/A')
        our_gap = f"{gap:.2f}" if gap is not None else "失败"
        diff = f"{gap - paper_gap:.2f}" if gap is not None and isinstance(paper_gap, float) else "N/A"
        lines.append(f"{name:<12} {paper_gap:<10} {our_gap:<10} {diff:<10}")

    # 打印到控制台
    print("\n" + "\n".join(lines))

    # 保存到文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n结果已保存到: {output_file}")
    except Exception as e:
        print(f"\n警告: 保存结果文件失败: {e}")


if __name__ == "__main__":
    main()