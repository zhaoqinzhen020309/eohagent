#!/usr/bin/env python3
"""
评估FSSP-GLS启发式算法在Taillard测试集上的性能
用法: python eval_fssp.py [--heuristic <heuristic_file>] [--n_jobs <num_processes>]
"""

import sys
import json
import numpy as np
import types
import time
import argparse
import os
from multiprocessing import Pool, cpu_count
from functools import partial

# 导入 Numba 编译的函数
from prob import makespan, local_search, local_search_perturb

# Taillard测试集配置 (对应TABLE 12)
TAILLARD_TEST_SETS = [
    ('t_j20_m5.txt', 20, 5),
    ('t_j20_m10.txt', 20, 10),
    ('t_j20_m20.txt', 20, 20),
    ('t_j50_m5.txt', 50, 5),
    ('t_j50_m10.txt', 50, 10),
    ('t_j50_m20.txt', 50, 20),
    ('t_j100_m5.txt', 100, 5),
    ('t_j100_m10.txt', 100, 10),
    ('t_j100_m20.txt', 100, 20),
    ('t_j200_m10.txt', 200, 10),
    ('t_j200_m20.txt', 200, 20),
]

def load_heuristic(heuristic_path='best_heuristic_fssp.py'):
    """加载启发式算法模块"""
    print(f"Loading heuristic: {heuristic_path}")

    try:
        with open(heuristic_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except FileNotFoundError:
        print(f"Error: File {heuristic_path} not found")
        print("Please run first: python extract_best_fssp.py <population_json>")
        sys.exit(1)

    # 创建算法模块
    algorithm_module = types.ModuleType("algorithm_module")
    try:
        exec(code, algorithm_module.__dict__)
    except Exception as e:
        print(f"Error: Code execution failed - {e}")
        sys.exit(1)

    # 验证函数存在
    if not hasattr(algorithm_module, 'get_matrix_and_jobs'):
        print("Error: Function 'get_matrix_and_jobs' not found in code")
        sys.exit(1)

    # 保存源代码以便多进程使用
    algorithm_module._source_code = code

    print("[OK] Heuristic loaded successfully\n")
    return algorithm_module

def read_taillard_file(filepath):
    """读取Taillard格式的测试文件，返回实例列表"""
    instances = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 寻找实例开始行
        if 'number of jobs' in line.lower():
            i += 1
            # 读取参数行
            params = lines[i].strip().split()
            n_jobs = int(params[0])
            n_machines = int(params[1])
            upper_bound = int(params[3])

            i += 1
            # 跳过 "processing times :" 行
            if 'processing' in lines[i].lower():
                i += 1

            # 读取处理时间矩阵 (n_machines行，每行n_jobs个值)
            time_matrix = []
            for _ in range(n_machines):
                if i >= len(lines):
                    break
                row_str = lines[i].strip()
                if row_str and not 'number of jobs' in row_str.lower():
                    row = list(map(int, row_str.split()))
                    time_matrix.append(row)
                    i += 1
                else:
                    break

            # 转换为numpy数组 (jobs × machines)
            time_matrix_np = np.array(time_matrix).T  # 转置：machines×jobs -> jobs×machines

            instances.append({
                'n_jobs': n_jobs,
                'n_machines': n_machines,
                'upper_bound': upper_bound,
                'time_matrix': time_matrix_np
            })
        else:
            i += 1

    return instances

def neh(tasks, machines_val, tasks_val):
    """NEH启发式初始化"""
    # 计算每个job的总处理时间
    total_times = [sum(tasks[j]) for j in range(tasks_val)]

    # 按总时间降序排列
    order = sorted(range(tasks_val), key=lambda x: total_times[x], reverse=True)

    # 构造序列
    current_seq = [order[0]]
    for i in range(1, tasks_val):
        best_seq = None
        min_cmax = float('inf')

        for pos in range(len(current_seq) + 1):
            temp_seq = current_seq[:]
            temp_seq.insert(pos, order[i])
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < min_cmax:
                min_cmax = cmax
                best_seq = temp_seq

        current_seq = best_seq

    return current_seq, makespan(current_seq, tasks, machines_val)

def run_gls(instance, heuristic, time_limit=60, iter_max=1000):
    """运行GLS算法"""
    # 设置随机种子以确保可重复性
    np.random.seed(2024)

    tasks = instance['time_matrix']
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']

    # NEH初始化
    pi, cmax = neh(tasks, n_machines, n_jobs)

    # GLS主循环
    pi_best = pi
    cmax_best = cmax
    n_itr = 0

    try:
        while n_itr < iter_max:
            # 局部搜索 - 与prob.py完全一致
            pi = local_search(pi, cmax, tasks, n_machines)
            cmax = makespan(pi, tasks, n_machines)  # 重新计算makespan

            if cmax < cmax_best:
                pi_best = pi
                cmax_best = cmax

            # 调用启发式更新矩阵和选择扰动jobs
            tasks_perturb, jobs = heuristic.get_matrix_and_jobs(pi, tasks.copy(), n_machines, n_jobs)

            # 验证jobs
            if len(jobs) <= 1:
                jobs = [0, 1]  # 至少两个job
            if len(jobs) > 5:
                jobs = jobs[:5]

            # 扰动局部搜索
            cmax = makespan(pi, tasks_perturb, n_machines)
            pi = local_search_perturb(pi, cmax, tasks_perturb, n_machines, jobs)

            n_itr += 1

            # 每50次迭代重置到最优解
            if n_itr % 50 == 0:
                pi = pi_best
                cmax = cmax_best

    except Exception as e:
        print(f"  [Warning] Exception during GLS: {e}")
        return cmax_best

    return cmax_best

def evaluate_single_instance(args):
    """评估单个实例（用于并行化）"""
    idx, instance, heuristic_code = args

    # 重新创建algorithm_module（因为不能在进程间共享）
    algorithm_module = types.ModuleType("algorithm_module")
    exec(heuristic_code, algorithm_module.__dict__)

    try:
        start_time = time.time()
        best_makespan = run_gls(instance, algorithm_module, time_limit=60, iter_max=1000)
        elapsed = time.time() - start_time

        gap = ((best_makespan - instance['upper_bound']) / instance['upper_bound']) * 100

        return {
            'idx': idx,
            'n_jobs': instance['n_jobs'],
            'n_machines': instance['n_machines'],
            'upper_bound': instance['upper_bound'],
            'best_makespan': best_makespan,
            'gap': gap,
            'time': elapsed,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'idx': idx,
            'n_jobs': instance['n_jobs'],
            'n_machines': instance['n_machines'],
            'success': False,
            'error': str(e)
        }

def evaluate_on_taillard(algorithm_module, n_jobs=None):
    """在Taillard测试集上评估算法（支持多进程）"""

    # 获取heuristic代码
    heuristic_code = getattr(algorithm_module, '_source_code', None)
    if heuristic_code is None:
        raise ValueError("Heuristic source code not found. Please reload the heuristic.")

    # 确定并行进程数
    if n_jobs is None:
        n_jobs = min(32, max(8, cpu_count() - 1))

    print("=" * 70)
    print("Start Evaluation on Taillard Test Set (TABLE 12)")
    print(f"Using {n_jobs} parallel processes (CPU count: {cpu_count()})")
    print("=" * 70)

    all_results = {}

    for test_file, expected_jobs, expected_machines in TAILLARD_TEST_SETS:
        test_name = f"{expected_jobs}×{expected_machines}"
        filepath = f'./TestingData/Taillard/{test_file}'

        if not os.path.exists(filepath):
            print(f"\n[SKIP] {test_name}: File not found - {filepath}")
            continue

        print(f"\n{'='*70}")
        print(f"Testing {test_name} ({test_file})")
        print(f"{'='*70}")

        # 读取实例
        instances = read_taillard_file(filepath)
        print(f"  Loaded {len(instances)} instances")

        # 准备参数列表
        eval_args = [(idx, inst, heuristic_code) for idx, inst in enumerate(instances)]

        # 并行评估
        print(f"  Evaluating {len(instances)} instances in parallel with {n_jobs} processes...")
        start_time_all = time.time()

        with Pool(processes=n_jobs) as pool:
            results = pool.map(evaluate_single_instance, eval_args)

        total_time = time.time() - start_time_all
        print(f"  Total evaluation time: {total_time:.1f}s")

        # 处理结果
        gaps = []
        for result in sorted(results, key=lambda x: x['idx']):
            idx = result['idx']
            if result['success']:
                print(f"    Instance {idx+1}/{len(instances)}: "
                      f"n={result['n_jobs']}, m={result['n_machines']}, "
                      f"UB={result['upper_bound']}, "
                      f"Makespan={result['best_makespan']:.0f}, "
                      f"Gap={result['gap']:.2f}%, "
                      f"Time={result['time']:.1f}s")
                gaps.append(result['gap'])
            else:
                print(f"    Instance {idx+1}/{len(instances)}: [ERROR] {result['error']}")

        # 计算平均gap
        if gaps:
            avg_gap = np.mean(gaps)
            all_results[test_name] = avg_gap
            print(f"\n  >>> Average Gap for {test_name}: {avg_gap:.2f}%")
        else:
            all_results[test_name] = None
            print(f"\n  >>> No valid results for {test_name}")

    return all_results

def print_results_table(results, output_file='results.txt'):
    """打印TABLE 12格式的结果并保存到文件"""

    test_order = [
        ('20×5', 20, 5), ('20×10', 20, 10), ('20×20', 20, 20),
        ('50×5', 50, 5), ('50×10', 50, 10), ('50×20', 50, 20),
        ('100×5', 100, 5), ('100×10', 100, 10), ('100×20', 100, 20),
        ('200×10', 200, 10), ('200×20', 200, 20),
    ]

    # 准备输出内容
    lines = []
    lines.append("=" * 70)
    lines.append("FSSP Taillard Test Set Evaluation Results (TABLE 12 Format)")
    lines.append("=" * 70)
    lines.append(f"{'Test Set':<15} {'Jobs':<8} {'Machines':<10} {'Avg Gap (%)':<12}")
    lines.append("-" * 70)

    valid_gaps = []
    for test_name, n_jobs, n_machines in test_order:
        gap = results.get(test_name)
        if gap is not None:
            gap_str = f"{gap:.2f}"
            valid_gaps.append(gap)
        else:
            gap_str = "N/A"
        lines.append(f"{test_name:<15} {n_jobs:<8} {n_machines:<10} {gap_str:<12}")

    if valid_gaps:
        avg_gap = np.mean(valid_gaps)
        lines.append("-" * 70)
        lines.append(f"{'AVERAGE':<15} {'':<8} {'':<10} {avg_gap:.2f}")

    lines.append("=" * 70)

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
    parser = argparse.ArgumentParser(description='评估FSSP-GLS启发式算法（支持多进程并行）')
    parser.add_argument('--heuristic', default='best_heuristic_fssp.py',
                       help='启发式算法文件路径 (默认: best_heuristic_fssp.py)')
    parser.add_argument('--n_jobs', type=int, default=None,
                       help=f'并行进程数 (默认: CPU核心数-1 = {max(1, cpu_count()-1)})')

    args = parser.parse_args()

    # 加载启发式算法
    algorithm_module = load_heuristic(args.heuristic)

    # 在Taillard上评估
    results = evaluate_on_taillard(algorithm_module, n_jobs=args.n_jobs)

    # 打印结果
    if results:
        print_results_table(results)
    else:
        print("\nEvaluation failed, no results obtained")
        sys.exit(1)

if __name__ == '__main__':
    main()