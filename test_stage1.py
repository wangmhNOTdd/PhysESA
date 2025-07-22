"""
阶段一快速测试脚本
用于验证数据预处理和模型加载是否正常工作
"""

import os
import sys
import torch
import pickle
import json
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "model"))

def test_data_preprocessing():
    """测试数据预处理"""
    print("=== 测试数据预处理 ===")
    
    # 检查必要文件
    data_root = "./datasets/pdbbind"
    if not os.path.exists(data_root):
        print(f"❌ 数据目录不存在: {data_root}")
        return False
    
    # 测试分子图构建器
    try:
        from molecular_graph import MolecularGraphBuilder
        builder = MolecularGraphBuilder(cutoff_radius=5.0, num_gaussians=16)
        print("✅ 分子图构建器初始化成功")
        
        # 测试特征维度
        dims = builder.get_feature_dimensions()
        print(f"✅ 特征维度: {dims}")
        
    except Exception as e:
        print(f"❌ 分子图构建器测试失败: {e}")
        return False
    
    # 测试简化版构建器
    try:
        from prepare_stage1_data import Stage1GraphBuilder
        stage1_builder = Stage1GraphBuilder(cutoff_radius=5.0, num_gaussians=16)
        dims = stage1_builder.get_feature_dimensions()
        print(f"✅ 阶段一图构建器: {dims}")
        
    except Exception as e:
        print(f"❌ 阶段一图构建器测试失败: {e}")
        return False
    
    return True


def test_model_loading():
    """测试ESA模型加载"""
    print("\n=== 测试ESA模型加载 ===")
    
    try:
        from esa.models import Estimator
        
        # 创建测试模型
        model = Estimator(
            task_type="regression",
            num_features=36,  # 10*2 + 16
            graph_dim=64,
            edge_dim=None,
            batch_size=2,
            lr=1e-4,
            linear_output_size=1,
            apply_attention_on="edge",
            layer_types=["M", "M"],
            hidden_dims=[64, 64],
            num_heads=4,
            set_max_items=100
        )
        
        print(f"✅ ESA模型创建成功")
        print(f"✅ 参数数量: {sum(p.numel() for p in model.parameters())}")
        
        return True
        
    except Exception as e:
        print(f"❌ ESA模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_format():
    """测试数据格式"""
    print("\n=== 测试数据格式 ===")
    
    # 检查是否存在预处理数据
    stage1_dir = "./experiments/stage1"
    data_files = ["train.pkl", "valid.pkl", "test.pkl", "metadata.json"]
    
    all_exist = True
    for file in data_files:
        file_path = os.path.join(stage1_dir, file)
        if os.path.exists(file_path):
            print(f"✅ 找到文件: {file}")
        else:
            print(f"❌ 缺少文件: {file}")
            all_exist = False
    
    if not all_exist:
        print("💡 请先运行数据预处理: python prepare_stage1_data.py --test_run")
        return False
    
    # 测试加载数据
    try:
        # 加载元数据
        with open(os.path.join(stage1_dir, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        print(f"✅ 元数据加载成功")
        print(f"   特征维度: {metadata['feature_dimensions']}")
        print(f"   数据集统计: {list(metadata['dataset_stats'].keys())}")
        
        # 加载一个样本测试
        with open(os.path.join(stage1_dir, "train.pkl"), 'rb') as f:
            train_data = pickle.load(f)
        
        if len(train_data) > 0:
            sample = train_data[0]
            print(f"✅ 训练数据加载成功，样本数: {len(train_data)}")
            print(f"   第一个样本:")
            print(f"   - 边表示形状: {sample['edge_representations'].shape}")
            print(f"   - 边数: {sample['num_edges']}")
            print(f"   - 节点数: {sample['num_nodes']}")
            print(f"   - 亲和力: {sample['affinity']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据格式测试失败: {e}")
        return False


def test_training_components():
    """测试训练组件"""
    print("\n=== 测试训练组件 ===")
    
    try:
        from experiments.stage1.train_stage1 import Stage1Dataset, collate_fn
        
        # 检查数据集类
        stage1_dir = "./experiments/stage1"
        train_file = os.path.join(stage1_dir, "train.pkl")
        
        if not os.path.exists(train_file):
            print("❌ 训练数据文件不存在，跳过训练组件测试")
            return False
        
        # 创建数据集
        dataset = Stage1Dataset(train_file)
        print(f"✅ 数据集创建成功，大小: {len(dataset)}")
        
        # 测试数据加载器
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # 测试一个批次
        batch = next(iter(dataloader))
        print(f"✅ 数据加载器测试成功")
        print(f"   批次键: {list(batch.keys())}")
        print(f"   x形状: {batch['x'].shape}")
        print(f"   y形状: {batch['y'].shape}")
        print(f"   最大边数: {batch['num_max_items']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🧪 阶段一系统测试")
    print("=" * 50)
    
    tests = [
        ("数据预处理", test_data_preprocessing),
        ("模型加载", test_model_loading), 
        ("数据格式", test_data_format),
        ("训练组件", test_training_components)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试出现异常: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("🏁 测试结果汇总")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有测试通过！可以开始阶段一训练")
        print("\n下一步:")
        print("1. 运行数据预处理（如果还没有）:")
        print("   python prepare_stage1_data.py --test_run")
        print("2. 开始训练:")
        print("   bash run_stage1.sh test")
    else:
        print("⚠️  存在测试失败，请检查环境配置")
        print("\n建议:")
        print("1. 确保已安装所有依赖")
        print("2. 确保PDBbind数据集已正确放置")
        print("3. 检查model/esa目录是否存在")


if __name__ == "__main__":
    main()
