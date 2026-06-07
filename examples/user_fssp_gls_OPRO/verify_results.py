#!/usr/bin/env python3
"""
验证评估结果是否合理
检查找到的Makespan是否低于Lower Bound（这是不可能的）
"""

import sys
import numpy as np
import types
import time
import os

# 复制必要的函数
def read_taillard_file_with_lb(filepath):
    """读取Taillard格式的测试文件，包括Lower Bound"""
    instances = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if 'number of jobs' in line.lower():
            i += 1
            params = lines[i].strip().split()
            n_jobs = int(params[0])
            n_machines = int(params[1])
            upper_bound = int(params[3])
            lower_bound = int(params[4])  # 读取Lower Bound

            i += 1
            if 'processing' in lines[i].lower():
                i += 1

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

            time_matrix_np = np.array(time_matrix).T

            instances.append({
                'n_jobs': n_jobs,
                'n_machines': n_machines,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound,
                'time_matrix': time_matrix_np
            })
        else:
            i += 1

    return instances

def makespan(order, tasks, machines_val):
    times = [0.0] * machines_val
    for j in order:
        times[0] += tasks[j][0]
        for k in range(1, machines_val):
            if times[k] < times[k-1]:
                times[k] = times[k-1]
            times[k] += tasks[j][k]
    return max(times)

def local_search(sequence, cmax_old, tasks, machines_val):
    new_seq = sequence[:]
    improved = True

    while improved:
        improved = False
        for i in range(len(new_seq)):
            for j in range(i+1, len(new_seq)):
                temp_seq = new_seq[:]
                temp_seq[i], temp_seq[j] = temp_seq[j], temp_seq[i]
                cmax = makespan(temp_seq, tasks, machines_val)
                if cmax < cmax_old:
                    new_seq = temp_seq[:]
                    cmax_old = cmax
                    improved = True
                    break
            if improved:
                break

        if not improved:
            for i in range(len(new_seq)):
                job = new_seq[i]
                for j in range(len(new_seq)):
                    if i != j:
                        temp_seq = new_seq[:]
                        temp_seq.pop(i)
                        temp_seq.insert(j, job)
                        cmax = makespan(temp_seq, tasks, machines_val)
                        if cmax < cmax_old:
                            new_seq = temp_seq[:]
                            cmax_old = cmax
                            improved = True
                            break
                if improved:
                    break

    return new_seq, cmax_old

def local_search_perturb(sequence, cmax_old, tasks, machines_val, jobs):
    new_seq = sequence[:]
    for i in jobs:
        for j in range(len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq.remove(i)
            temp_seq.insert(j, i)
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax
    return new_seq

def neh(tasks, machines_val, tasks_val):
    total_times = [sum(tasks[j]) for j in range(tasks_val)]
    order = sorted(range(tasks_val), key=lambda x: total_times[x], reverse=True)
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
    tasks = instance['time_matrix']
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']

    pi, cmax = neh(tasks, n_machines, n_jobs)
    pi_best = pi
    cmax_best = cmax
    n_itr = 0
    time_start = time.time()

    try:
        while time.time() - time_start < time_limit and n_itr < iter_max:
            pi, cmax = local_search(pi, cmax, tasks, n_machines)

            if cmax < cmax_best:
                pi_best = pi
                cmax_best = cmax

            tasks_perturb, jobs = heuristic.get_matrix_and_jobs(pi, tasks.copy(), n_machines, n_jobs)

            if len(jobs) <= 1:
                jobs = [0, 1]
            if len(jobs) > 5:
                jobs = jobs[:5]

            cmax = makespan(pi, tasks_perturb, n_machines)
            pi = local_search_perturb(pi, cmax, tasks_perturb, n_machines, jobs)

            n_itr += 1

            if n_itr % 50 == 0:
                pi = pi_best
                cmax = cmax_best

    except Exception as e:
        print(f"  [Warning] Exception during GLS: {e}")
        return cmax_best, pi_best

    return cmax_best, pi_best

def main():
    # 加载启发式
    with open('best_heuristic_fssp.py', 'r', encoding='utf-8') as f:
        code = f.read()

    algorithm_module = types.ModuleType("algorithm_module")
    exec(code, algorithm_module.__dict__)

    # 读取20×5测试集
    filepath = './TestingData/Taillard/t_j20_m5.txt'
    instances = read_taillard_file_with_lb(filepath)

    print("=" * 80)
    print("验证 OPRO 在 20×5 测试集上的结果")
    print("=" * 80)
    print(f"{'Inst':<6} {'Jobs':<6} {'Mach':<6} {'LB':<8} {'UB':<8} {'Found':<8} {'Gap%':<10} {'Valid'}")
    print("-" * 80)

    gaps = []
    all_valid = True

    for idx, instance in enumerate(instances):
        best_makespan, best_seq = run_gls(instance, algorithm_module, time_limit=60, iter_max=1000)

        lb = instance['lower_bound']
        ub = instance['upper_bound']
        gap = ((best_makespan - ub) / ub) * 100
        gaps.append(gap)

        # 检查是否低于Lower Bound
        valid = "✓" if best_makespan >= lb else "✗ INVALID!"
        if best_makespan < lb:
            all_valid = False

        print(f"{idx+1:<6} {instance['n_jobs']:<6} {instance['n_machines']:<6} "
              f"{lb:<8} {ub:<8} {best_makespan:<8.0f} {gap:<10.2f} {valid}")

    avg_gap = np.mean(gaps)
    print("-" * 80)
    print(f"平均 Gap: {avg_gap:.2f}%")
    print("=" * 80)

    if all_valid:
        print("\n✓ 所有结果都有效（Makespan >= Lower Bound）")
    else:
        print("\n✗ 警告：存在无效结果（Makespan < Lower Bound）！")
        print("  这表明评估代码或启发式存在问题。")

if __name__ == '__main__':
    main()