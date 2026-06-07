"""
BPPLib数据加载器 - 用于加载和处理BPPLib标准测试集
"""
import pickle
import numpy as np
from pathlib import Path


class GetBPPLibData:
    """加载BPPLib数据集"""
    
    def __init__(self, data_dir='bpplib_converted'):
        self.data_dir = Path(data_dir)
    
    def get_instances(self, capacity, dataset_name):
        """
        获取BPPLib数据集实例
        
        Args:
            capacity: 箱子容量（用于评估，忽略原始数据中的capacity）
            dataset_name: 数据集名称，例如 'Falkenauer_U_120', 'Scholl_50'
        
        Returns:
            instances: {instance_name: {'capacity': c, 'num_items': n, 'items': [...]}}
            lb: {instance_name: lower_bound} - 理论下界
        """
        pkl_file = self.data_dir / f"{dataset_name}.pkl"
        
        if not pkl_file.exists():
            raise FileNotFoundError(f"数据集文件不存在: {pkl_file}")
        
        # 加载数据
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        # 数据格式: {num_items: [instance1, instance2, ...]}
        num_items = list(data.keys())[0]
        instances_list = data[num_items]
        
        # 构造返回格式 - 与evaluation.py期望的格式一致
        instances = {}
        lb = {}
        
        for idx, items in enumerate(instances_list):
            instance_name = f"{dataset_name}_test_{idx+1}"
            
            # 构造实例字典
            instances[instance_name] = {
                'capacity': capacity,
                'num_items': len(items),
                'items': items
            }
            
            # 计算理论下界
            total_size = sum(items)
            lower_bound = np.ceil(total_size / capacity)
            lb[instance_name] = lower_bound
        
        return instances, lb
    
    def get_all_datasets(self):
        """获取所有可用的数据集名称"""
        pkl_files = list(self.data_dir.glob('*.pkl'))
        dataset_names = [f.stem for f in pkl_files]
        return sorted(dataset_names)


if __name__ == "__main__":
    # 测试
    loader = GetBPPLibData()
    
    print("可用的数据集:")
    datasets = loader.get_all_datasets()
    for ds in datasets:
        print(f"  - {ds}")
    
    print("\n测试加载 Falkenauer_U_120:")
    instances, lb = loader.get_instances(capacity=150, dataset_name='Falkenauer_U_120')
    
    print(f"数据集数量: {len(instances)}")
    print(f"实例数量: {len(instances['Falkenauer_U_120'])}")
    print(f"第一个实例物品数: {len(instances['Falkenauer_U_120'][0])}")
    print(f"下界示例: {list(lb.items())[:3]}")

