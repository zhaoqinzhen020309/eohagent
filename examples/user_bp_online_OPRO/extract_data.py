import os
import json
import re

# 配置您的文件夹路径
base_path = r"C:\Users\48520\Desktop\eoh多目标\examples\user_bp_online_OPRO"
folders = ['pop10', 'pop20', 'pop30']


def get_convergence_data(folder_name):
    folder_full_path = os.path.join(base_path, folder_name, 'results', 'pops')
    if not os.path.exists(folder_full_path):
        print(f"Warning: Folder not found: {folder_full_path}")
        return []

    # 获取所有json文件
    files = [f for f in os.listdir(folder_full_path) if f.startswith('population_generation_') and f.endswith('.json')]

    # 按代数排序 (1, 2, ..., 10, 11...)
    files.sort(key=lambda x: int(re.search(r'(\d+)', x).group(1)))

    best_values = []
    iterations = []

    for f in files:
        gen_num = int(re.search(r'(\d+)', f).group(1))
        file_path = os.path.join(folder_full_path, f)

        try:
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                # 提取当前代所有个体的 objective 值
                objectives = [item['objective'] for item in data if
                              'objective' in item and item['objective'] is not None]

                if objectives:
                    # 找到这一代的最优值（最小值）
                    best_val = min(objectives)
                    # 如果数据是小数（如0.029），转换为百分比（如2.928）
                    # 假设您的objective是小数形式，这里乘以100转为%
                    # 如果原本就是百分比，请去掉 * 100
                    best_values.append(best_val * 100)
                    iterations.append(gen_num)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    return iterations, best_values


# 执行并打印结果
print("-" * 30)
for folder in folders:
    iters, values = get_convergence_data(folder)
    print(f"\nProcessing {folder}...")
    # 格式化输出，方便您直接复制
    print(f"{folder}_x = {iters}")
    print(f"{folder}_y = {['{:.3f}'.format(v) for v in values]}")
print("-" * 30)