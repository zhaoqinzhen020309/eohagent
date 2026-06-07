#!/usr/bin/env python3
"""
提取TSP-GLS最佳启发式算法
用法: python extract_best_tsp.py <population_json_path> --output <output_file>
"""

import json
import argparse
import sys

def extract_best_heuristic(json_path, output_path='best_heuristic.py'):
    """从种群JSON文件中提取最佳个体的代码"""
    
    print("=" * 60)
    print("Extract Best TSP-GLS Heuristic")
    print("=" * 60)
    
    # 读取种群文件
    print(f"Reading file: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {json_path} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: JSON parsing failed - {e}")
        sys.exit(1)
    
    # 处理单个个体或种群列表
    if isinstance(data, list):
        # 种群列表，取第一个（最佳）
        if len(data) == 0:
            print("Error: Population is empty")
            sys.exit(1)
        best_individual = data[0]
    elif isinstance(data, dict):
        # 单个个体
        best_individual = data
    else:
        print(f"Error: Unknown data format")
        sys.exit(1)
    
    # 提取信息
    fitness = best_individual.get('objective', 'N/A')
    algorithm_desc = best_individual.get('algorithm', 'No description')
    code = best_individual.get('code', '')
    
    if not code:
        print("Error: Code is empty")
        sys.exit(1)
    
    print(f"Best individual fitness: {fitness}")
    print(f"Algorithm description: {algorithm_desc}")
    print()
    print("Best individual code:")
    print("-" * 50)
    print(code)
    print("-" * 50)
    
    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(code)
    
    print(f"\nSaved to: {output_path}")
    print("\nExtraction complete!")
    print("Now you can run: python eval_tsp.py")

def main():
    parser = argparse.ArgumentParser(description='提取TSP-GLS最佳启发式算法')
    parser.add_argument('json_path', help='种群JSON文件路径')
    parser.add_argument('--output', default='best_heuristic.py', help='输出文件路径 (默认: best_heuristic.py)')
    
    args = parser.parse_args()
    extract_best_heuristic(args.json_path, args.output)

if __name__ == '__main__':
    main()

