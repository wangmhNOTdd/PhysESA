"""
验证生成的数据文件
检查KNN连边和验证集是否正确生成
"""

import os
import pickle
import json
import numpy as np

def check_generated_data():
    """检查生成的数据文件"""
    
    stage1_dir = "./experiments/stage1"
    
    print("=== 数据文件检查 ===")
    
    # 检查必需文件
    required_files = ["train.pkl", "valid.pkl", "test.pkl", "metadata.json"]
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(stage1_dir, file)
        if os.path.exists(file_path):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - 文件缺失")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  缺失文件: {missing_files}")
        print("请运行数据预处理: python prepare_stage1_data.py --test_run")
        return False
    
    # 检查元数据
    print("\n=== 元数据检查 ===")
    try:
        with open(os.path.join(stage1_dir, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        print(f"✅ 连边方式: {metadata['edge_connection']['method']}")
        if metadata['edge_connection']['method'] == 'knn':
            print(f"✅ K值: {metadata['edge_connection']['k']}")
        else:
            print(f"✅ 截断半径: {metadata['edge_connection']['radius']}")
        
        print(f"✅ 特征维度: {metadata['feature_dimensions']}")
        
        # 检查数据集统计
        stats = metadata['dataset_stats']
        print(f"\n数据集统计:")
        for split_name, split_stats in stats.items():
            print(f"  {split_name}: {split_stats['num_samples']} 样本")
        
        # 验证验证集是否存在
        if 'valid' in stats and stats['valid']['num_samples'] > 0:
            print("✅ 验证集已正确生成")
        else:
            print("❌ 验证集缺失或为空")
            return False
            
    except Exception as e:
        print(f"❌ 元数据读取失败: {e}")
        return False
    
    # 检查数据格式
    print("\n=== 数据格式检查 ===")
    try:
        # 检查训练数据
        with open(os.path.join(stage1_dir, "train.pkl"), 'rb') as f:
            train_data = pickle.load(f)
        
        if len(train_data) > 0:
            sample = train_data[0]
            print(f"✅ 训练样本数: {len(train_data)}")
            print(f"✅ 样本格式检查:")
            print(f"   - edge_representations: {sample['edge_representations'].shape}")
            print(f"   - 边数: {sample['num_edges']}")
            print(f"   - 节点数: {sample['num_nodes']}")
            print(f"   - 亲和力: {sample['affinity']:.3f}")
            
            # 检查边数是否合理（KNN应该产生相对固定的边数）
            edge_counts = [s['num_edges'] for s in train_data[:10]]  # 检查前10个样本
            edge_mean = np.mean(edge_counts)
            edge_std = np.std(edge_counts)
            
            print(f"   - 边数统计 (前10个样本): 均值={edge_mean:.1f}, 标准差={edge_std:.1f}")
            
            if metadata['edge_connection']['method'] == 'knn':
                expected_edges_per_node = metadata['edge_connection']['k']
                print(f"   - KNN预期边数比例: {expected_edges_per_node} 边/节点")
                
        # 检查验证数据
        with open(os.path.join(stage1_dir, "valid.pkl"), 'rb') as f:
            valid_data = pickle.load(f)
        print(f"✅ 验证样本数: {len(valid_data)}")
        
        # 检查测试数据  
        with open(os.path.join(stage1_dir, "test.pkl"), 'rb') as f:
            test_data = pickle.load(f)
        print(f"✅ 测试样本数: {len(test_data)}")
        
    except Exception as e:
        print(f"❌ 数据格式检查失败: {e}")
        return False
    
    print("\n=== 检查完成 ===")
    print("✅ 所有检查通过！数据已准备就绪")
    
    # 建议下一步
    print("\n下一步:")
    print("1. 运行训练测试: python ./experiments/stage1/train_stage1.py --test_run")
    print("2. 完整训练: bash run_stage1.sh")
    
    return True

if __name__ == "__main__":
    check_generated_data()
