import os
import json
import pandas as pd
import re


def extract_best_fitness(pop_dir):
    """
    扫描目录下的所有 population_generation_X.json 文件，提取每一代的最优 fitness。
    """
    results = {}
    if not os.path.exists(pop_dir):
        print(f"警告: 找不到目录 {pop_dir}")
        return results

    # 遍历文件夹下的所有 json 文件
    for filename in os.listdir(pop_dir):
        if filename.endswith(".json") and "population_generation" in filename:
            # 提取代数数字
            gen_match = re.search(r'generation_(\d+)', filename)
            if gen_match:
                gen = int(gen_match.group(1))
                file_path = os.path.join(pop_dir, filename)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        # 假设 JSON 是一个列表，每个元素有个体信息
                        # 兼容不同版本的保存格式 (可能有 "objective" 或 "fitness" 键)
                        if isinstance(data, list):
                            # 提取所有个体的目标值（通常是最小化问题）
                            objectives = [ind.get('objective', ind.get('fitness', 999)) for ind in data]
                            best_val = min(objectives)
                            results[gen] = best_val
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")

    return results


def main():
    # 根据您的目录结构设置路径
    base_dirs = {
        "N10": "./pop10/results/pops",  # 请根据实际情况微调路径
        "N20": "./pop20/results/pops",
        "N30": "./pop30/results/pops"
    }

    all_data = []

    for label, path in base_dirs.items():
        print(f"正在处理 {label} 数据...")
        gen_data = extract_best_fitness(path)
        for gen, val in gen_data.items():
            all_data.append({
                "Generation": gen,
                "N_Value": label,
                "Optimality_Gap": val
            })

    # 转换为 DataFrame
    df = pd.DataFrame(all_data)

    if df.empty:
        print("未提取到任何数据，请检查路径是否正确。")
        return

    # 透视表格式：行是代数，列是不同的 N 值
    pivot_df = df.pivot(index="Generation", columns="N_Value", values="Optimality_Gap").sort_index()

    # 导出 CSV
    output_file = "evolution_raw_data.csv"
    pivot_df.to_csv(output_file)
    print(f"\n成功！原始数据已导出至: {output_file}")
    print("\n前 5 代数据预览：")
    print(pivot_df.head())


if __name__ == "__main__":
    main()