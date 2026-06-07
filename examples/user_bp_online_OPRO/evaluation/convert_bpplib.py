"""
完整处理BPPLib数据集
将BPPLib的RAR文件解压并转换为 {num_items: [instance1, instance2, ...]} 格式
"""
import os
import zipfile
import rarfile
import pickle
import numpy as np
from pathlib import Path


class BPPLibConverter:
    """BPPLib数据集转换器"""
    
    def __init__(self, bpplib_dir, output_dir='bpplib_converted'):
        self.bpplib_dir = Path(bpplib_dir)
        self.benchmarks_dir = self.bpplib_dir / 'Instances' / 'Benchmarks'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 临时解压目录
        self.temp_dir = self.output_dir / 'temp_extracted'
        self.temp_dir.mkdir(exist_ok=True)
    
    def extract_rar_file(self, rar_path, extract_to):
        """解压RAR文件"""
        try:
            with rarfile.RarFile(rar_path) as rf:
                rf.extractall(extract_to)
            return True
        except Exception as e:
            print(f"解压失败 {rar_path.name}: {e}")
            return False
    
    def parse_bpp_file(self, filepath):
        """
        解析BPP格式文件
        格式:
        第1行: n (物品数量)
        第2行: c (箱子容量)
        接下来n行: 每行一个物品大小
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            if len(lines) < 2:
                return None
            
            n = int(lines[0])
            c = int(lines[1])
            
            items = []
            for i in range(2, min(2 + n, len(lines))):
                try:
                    items.append(int(lines[i]))
                except:
                    continue
            
            if len(items) != n:
                print(f"  警告: {filepath.name} 期望{n}个物品，实际读取{len(items)}个")
            
            return {
                'capacity': c,
                'num_items': len(items),
                'items': items
            }
        
        except Exception as e:
            print(f"  解析失败 {filepath.name}: {e}")
            return None
    
    def process_rar_file(self, rar_filename):
        """处理单个RAR文件"""
        rar_path = self.benchmarks_dir / rar_filename
        
        if not rar_path.exists():
            print(f"文件不存在: {rar_filename}")
            return None
        
        print(f"\n处理: {rar_filename}")
        print("-" * 80)
        
        # 解压到临时目录
        extract_dir = self.temp_dir / rar_filename.replace('.rar', '')
        extract_dir.mkdir(exist_ok=True)
        
        print(f"解压中...")
        if not self.extract_rar_file(rar_path, extract_dir):
            return None
        
        # 查找所有txt文件
        txt_files = list(extract_dir.rglob('*.txt'))
        print(f"找到 {len(txt_files)} 个txt文件")
        
        # 解析所有文件
        instances_by_size = {}  # {num_items: [items_list1, items_list2, ...]}
        
        for txt_file in txt_files:
            instance = self.parse_bpp_file(txt_file)
            
            if instance:
                num_items = instance['num_items']
                
                if num_items not in instances_by_size:
                    instances_by_size[num_items] = []
                
                instances_by_size[num_items].append(instance['items'])
        
        # 保存为pkl文件
        category_name = rar_filename.replace('.rar', '').replace('_BPP', '').replace('_CSP', '')
        
        for num_items, items_lists in sorted(instances_by_size.items()):
            # 格式: {num_items: [items_list1, items_list2, ...]}
            data = {num_items: items_lists}
            
            # 文件名: category_numitems.pkl
            output_filename = f"{category_name}_{num_items}.pkl"
            output_path = self.output_dir / output_filename
            
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"  ✓ {output_filename}: {len(items_lists)} 个实例, {num_items} 个物品")
        
        return instances_by_size
    
    def process_all_recommended(self):
        """处理推荐的数据集"""
        print("=" * 80)
        print("处理BPPLib推荐数据集")
        print("=" * 80)
        
        # 推荐的数据集（BPP格式）
        recommended = [
            '1_Falkenauer.rar',      # Uniform和Triplets分布
            '2_Scholl.rar',          # 经典测试集
            '5_Hard28.rar',          # 困难实例
            '6_Randomly_Generated.rar'  # 随机生成
        ]
        
        results = {}
        
        for rar_file in recommended:
            result = self.process_rar_file(rar_file)
            if result:
                results[rar_file] = result
        
        print()
        print("=" * 80)
        print("处理完成")
        print("=" * 80)
        
        # 生成摘要
        self.generate_summary()
        
        return results
    
    def generate_summary(self):
        """生成摘要"""
        pkl_files = sorted(self.output_dir.glob('*.pkl'))
        
        print(f"\n生成的文件 (共 {len(pkl_files)} 个):")
        print("-" * 80)
        
        total_instances = 0
        
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            for num_items, items_lists in data.items():
                num_instances = len(items_lists)
                total_instances += num_instances
                
                # 计算统计信息
                all_items = [item for items in items_lists for item in items]
                
                print(f"{pkl_file.name:40} | 物品数: {num_items:6} | 实例数: {num_instances:4} | "
                      f"物品范围: [{min(all_items)}, {max(all_items)}]")
        
        print("-" * 80)
        print(f"总计: {len(pkl_files)} 个文件, {total_instances} 个实例")
        print()
        print(f"所有文件已保存到: {self.output_dir.absolute()}")
        print("=" * 80)


def scan_extracted_files(extracted_dir, output_dir='bpplib_converted'):
    """扫描已解压的文件并转换（手动解压模式）"""
    from pathlib import Path
    
    extracted_path = Path(extracted_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("扫描已解压的BPPLib文件")
    print("=" * 80)
    print(f"扫描目录: {extracted_path}")
    print()
    
    # 递归查找所有txt文件
    txt_files = list(extracted_path.rglob('*.txt'))
    
    if not txt_files:
        print("❌ 未找到任何txt文件！")
        print("请确保已经手动解压了RAR文件")
        return
    
    print(f"找到 {len(txt_files)} 个txt文件")
    print()
    
    # 按类别和大小组织
    data_by_category = {}
    
    converter = BPPLibConverter('', output_dir)
    
    for txt_file in txt_files:
        instance = converter.parse_bpp_file(txt_file)
        
        if instance:
            # 推断类别
            path_str = str(txt_file).lower()
            if 'falkenauer' in path_str and '_u' in path_str:
                category = 'Falkenauer_U'
            elif 'falkenauer' in path_str and '_t' in path_str:
                category = 'Falkenauer_T'
            elif 'scholl' in path_str:
                category = 'Scholl'
            elif 'hard' in path_str:
                category = 'Hard28'
            else:
                category = 'Other'
            
            num_items = instance['num_items']
            
            if category not in data_by_category:
                data_by_category[category] = {}
            if num_items not in data_by_category[category]:
                data_by_category[category][num_items] = []
            
            data_by_category[category][num_items].append(instance['items'])
    
    # 保存
    print("转换为pkl格式...")
    print("-" * 80)
    
    total_files = 0
    total_instances = 0
    
    for category, size_data in sorted(data_by_category.items()):
        for num_items, items_lists in sorted(size_data.items()):
            data = {num_items: items_lists}
            
            output_filename = f"{category}_{num_items}.pkl"
            output_filepath = output_path / output_filename
            
            with open(output_filepath, 'wb') as f:
                pickle.dump(data, f)
            
            num_instances = len(items_lists)
            total_files += 1
            total_instances += num_instances
            
            all_items = [item for items in items_lists for item in items]
            print(f"✓ {output_filename:40} | 物品数: {num_items:6} | 实例数: {num_instances:4} | "
                  f"范围: [{min(all_items)}, {max(all_items)}]")
    
    print("-" * 80)
    print(f"总计: {total_files} 个文件, {total_instances} 个实例")
    print(f"所有文件已保存到: {output_path.absolute()}")
    print("=" * 80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BPPLib数据集转换工具')
    parser.add_argument('--extracted_dir', type=str, 
                        default='../../../BPPLIB-main/extracted',
                        help='已解压的BPPLib目录路径')
    parser.add_argument('--output_dir', type=str, 
                        default='bpplib_converted',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 直接扫描已解压的文件
    scan_extracted_files(args.extracted_dir, args.output_dir)
    
    print()
    print("✓ 完成！")
    print()
    print("下一步:")
    print("  1. 查看 bpplib_converted/ 目录中的pkl文件")
    print("  2. 这些文件格式与 testingdata/ 完全相同")
    print("  3. 可以直接用 runEval.py 测试")


if __name__ == "__main__":
    main()

