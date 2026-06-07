"""
BPPLib评估脚本 - 用于测试启发式在BPPLib标准测试集上的性能
"""
import importlib
from get_bpplib_instance import GetBPPLibData
from evaluation import Evaluation

eva = Evaluation()

# BPPLib数据集分组 - 根据实际物品大小范围配置合适的capacity
# 基于check_bpplib_data.py的分析结果

# Falkenauer_U系列: 物品[20,100], 平均60
# 使用C=100，与训练集保持一致，便于公平对比
datasets_U = [
    'Falkenauer_U_120',    # 20个实例，每个120个物品
    'Falkenauer_U_250',    # 20个实例，每个250个物品
    'Falkenauer_U_500',    # 20个实例，每个500个物品
    'Falkenauer_U_1000',   # 20个实例，每个1000个物品
]

# Falkenauer_T系列: 物品[250,499], 平均333
# Triplets结构：1大物品+3小物品，最优解是每箱装1组
# 使用C=500，刚好能装下最大物品
datasets_T = [
    'Falkenauer_T_60',     # 20个实例，每个60个物品
    'Falkenauer_T_120',    # 20个实例，每个120个物品
    'Falkenauer_T_249',    # 20个实例，每个249个物品
    'Falkenauer_T_501',    # 20个实例，每个501个物品
]

# Scholl系列: 物品[1,627], 平均113
# 经典BPP测试集，包含各种困难实例
# 使用C=700，稍大于最大物品
datasets_Scholl = [
    'Scholl_50',           # 300个实例，每个50个物品
    'Scholl_100',          # 300个实例，每个100个物品
    'Scholl_500',          # 300个实例，每个500个物品
]

# Hard28系列: 物品[1,800], 平均380-390
# 专门设计的困难实例
# 使用C=800，刚好能装下最大物品
datasets_Hard28 = [
    'Hard28_160',          # 8个实例，每个160个物品
    'Hard28_180',          # 10个实例，每个180个物品
    'Hard28_200',          # 10个实例，每个200个物品
]

# 测试配置: (数据集列表, capacity, 描述)
test_configs = [
    (datasets_U, 100, "Falkenauer_U (Uniform均匀分布, 物品20-100)"),
    (datasets_T, 500, "Falkenauer_T (Triplets三元组, 物品250-499)"),
    (datasets_Scholl, 700, "Scholl (经典测试集, 物品1-627)"),
    (datasets_Hard28, 800, "Hard28 (困难实例, 物品1-800)"),
]

print("=" * 80)
print("BPPLib标准测试集评估")
print("=" * 80)
print()

with open("results_bpplib.txt", "w", encoding='utf-8') as file:
    file.write("BPPLib标准测试集评估结果\n")
    file.write("=" * 80 + "\n\n")
    
    # 按数据集分组测试
    for datasets, capacity, description in test_configs:
        print(f"\n{'='*80}")
        print(f"{description} | Capacity={capacity}")
        print(f"{'='*80}")
        file.write(f"\n{description} | Capacity={capacity}\n")
        file.write("=" * 80 + "\n")
        
        for dataset_name in datasets:
            try:
                # 加载数据
                getdata = GetBPPLibData()
                instances, lb = getdata.get_instances(capacity, dataset_name)
                
                # 重新加载启发式（确保使用最新版本）
                heuristic_module = importlib.import_module("heuristic")
                heuristic = importlib.reload(heuristic_module)
                
                # 评估
                avg_num_bins = -eva.evaluateGreedy(instances, heuristic)
                
                # 计算平均下界
                avg_lb = sum(lb.values()) / len(lb)
                
                # 计算excess
                excess = (avg_num_bins - avg_lb) / avg_lb
                
                result = f'{dataset_name:25} | Bins: {avg_num_bins:7.2f} | LB: {avg_lb:7.2f} | Excess: {100 * excess:6.2f}%'
                print(result)
                file.write(result + "\n")
                
            except FileNotFoundError:
                msg = f'{dataset_name:25} | 数据集不存在'
                print(msg)
                file.write(msg + "\n")
            except Exception as e:
                msg = f'{dataset_name:25} | 错误: {str(e)}'
                print(msg)
                file.write(msg + "\n")

print()
print("=" * 80)
print("评估完成！结果已保存到 results_bpplib.txt")
print("=" * 80)

