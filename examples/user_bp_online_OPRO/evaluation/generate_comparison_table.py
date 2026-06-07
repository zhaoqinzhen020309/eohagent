"""
生成BPPLib测试结果的表格
"""

# 训练集结果（Weibull分布）
weibull_results = {
    100: {'1k': 3.93, '5k': 0.85, '10k': 0.42},
    200: {'1k': 3.65, '5k': 0.82, '10k': 0.39},
    300: {'1k': 3.67, '5k': 0.83, '10k': 0.39},
    400: {'1k': 3.68, '5k': 0.86, '10k': 0.40},
    500: {'1k': 3.67, '5k': 0.85, '10k': 0.40},
}

# BPPLib测试集结果
bpplib_results = {
    'Falkenauer_U': {
        'capacity': 100,
        'item_range': '[20,100]',
        'results': {
            '120': 42.24,
            '250': 39.53,
            '500': 37.29,
            '1000': 35.88,
        }
    },
    'Falkenauer_T': {
        'capacity': 500,
        'item_range': '[250,499]',
        'results': {
            '60': 49.12,
            '120': 48.25,
            '249': 48.13,
            '501': 48.40,
        }
    },
    'Scholl': {
        'capacity': 700,
        'item_range': '[1,627]',
        'results': {
            '50': 112.32,
            '100': 112.79,
            '500': 111.31,
        }
    },
    'Hard28': {
        'capacity': 800,
        'item_range': '[1,800]',
        'results': {
            '160': 54.07,
            '180': 50.79,
            '200': 52.78,
        }
    }
}

print("=" * 120)
print("训练集 vs 测试集性能对比表")
print("=" * 120)
print()

# 表格1: 训练集（Weibull分布）
print("【训练集 - Weibull分布】")
print("-" * 120)
print(f"{'Capacity':<12} | {'Size':<8} | {'Excess%':<10} | 说明")
print("-" * 120)

for capacity in [100, 200, 300, 400, 500]:
    results = weibull_results[capacity]
    for i, (size, excess) in enumerate(results.items()):
        if i == 0:
            print(f"{capacity:<12} | {size:<8} | {excess:<10.2f} | 物品大小1-100 (Weibull分布)")
        else:
            print(f"{'':12} | {size:<8} | {excess:<10.2f} |")

# 计算平均
avg_excess = sum([v for cap_results in weibull_results.values() for v in cap_results.values()]) / 15
print("-" * 120)
print(f"{'平均':<12} | {'':8} | {avg_excess:<10.2f} |")
print()

# 表格2: 测试集（BPPLib）
print("=" * 120)
print("【测试集 - BPPLib标准测试集】")
print("-" * 120)
print(f"{'数据集':<20} | {'Capacity':<10} | {'物品范围':<15} | {'Size':<8} | {'Excess%':<10} | 说明")
print("-" * 120)

for dataset_name, dataset_info in bpplib_results.items():
    capacity = dataset_info['capacity']
    item_range = dataset_info['item_range']
    results = dataset_info['results']
    
    # 数据集说明
    if dataset_name == 'Falkenauer_U':
        desc = "Uniform均匀分布"
    elif dataset_name == 'Falkenauer_T':
        desc = "Triplets三元组 (1大+3小)"
    elif dataset_name == 'Scholl':
        desc = "经典困难测试集"
    elif dataset_name == 'Hard28':
        desc = "专门设计的困难实例"
    
    for i, (size, excess) in enumerate(results.items()):
        if i == 0:
            print(f"{dataset_name:<20} | {capacity:<10} | {item_range:<15} | {size:<8} | {excess:<10.2f} | {desc}")
        else:
            print(f"{'':20} | {'':10} | {'':15} | {size:<8} | {excess:<10.2f} |")
    
    # 计算该数据集平均
    avg = sum(results.values()) / len(results)
    print(f"{'  └─ 平均':<20} | {'':10} | {'':15} | {'':8} | {avg:<10.2f} |")
    print("-" * 120)

# 总体平均
all_excess = [v for ds in bpplib_results.values() for v in ds['results'].values()]
overall_avg = sum(all_excess) / len(all_excess)
print(f"{'总体平均':<20} | {'':10} | {'':15} | {'':8} | {overall_avg:<10.2f} |")
print()

# 对比分析
print("=" * 120)
print("【过拟合分析】")
print("=" * 120)
print(f"训练集平均Excess:  {avg_excess:.2f}%")
print(f"测试集平均Excess:  {overall_avg:.2f}%")
print(f"性能下降:          {overall_avg - avg_excess:.2f}% (绝对值)")
print(f"性能下降倍数:      {overall_avg / avg_excess:.2f}x")
print()

# 详细对比
print("=" * 120)
print("【详细对比 - 按数据集】")
print("=" * 120)
print(f"{'数据集':<20} | {'训练集Excess':<15} | {'测试集Excess':<15} | {'性能下降':<12} | 结论")
print("-" * 120)

# Falkenauer_U vs Weibull (都是C=100)
weibull_100_avg = sum(weibull_results[100].values()) / len(weibull_results[100])
falkenauer_u_avg = sum(bpplib_results['Falkenauer_U']['results'].values()) / len(bpplib_results['Falkenauer_U']['results'])
print(f"{'Falkenauer_U':<20} | {weibull_100_avg:<15.2f} | {falkenauer_u_avg:<15.2f} | {falkenauer_u_avg/weibull_100_avg:<12.2f}x | 同capacity(100)，不同分布")

# Falkenauer_T
falkenauer_t_avg = sum(bpplib_results['Falkenauer_T']['results'].values()) / len(bpplib_results['Falkenauer_T']['results'])
weibull_500_avg = sum(weibull_results[500].values()) / len(weibull_results[500])
print(f"{'Falkenauer_T':<20} | {weibull_500_avg:<15.2f} | {falkenauer_t_avg:<15.2f} | {falkenauer_t_avg/weibull_500_avg:<12.2f}x | Triplets结构，算法无法识别")

# Scholl
scholl_avg = sum(bpplib_results['Scholl']['results'].values()) / len(bpplib_results['Scholl']['results'])
print(f"{'Scholl':<20} | {avg_excess:<15.2f} | {scholl_avg:<15.2f} | {scholl_avg/avg_excess:<12.2f}x | 经典困难实例")

# Hard28
hard28_avg = sum(bpplib_results['Hard28']['results'].values()) / len(bpplib_results['Hard28']['results'])
print(f"{'Hard28':<20} | {avg_excess:<15.2f} | {hard28_avg:<15.2f} | {hard28_avg/avg_excess:<12.2f}x | 专门设计的困难实例")

print("=" * 120)

