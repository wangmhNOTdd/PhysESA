#!/usr/bin/env python3
"""
快速测试脚本 - 验证阶段一模型是否能正常工作
"""

import torch
import sys
import os

# 添加src到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.molecular_graph import MolecularGraphBuilder, test_molecular_graph
from src.models.esa_components import test_esa_components
from src.models.physesa import create_model_stage1, test_model
from src.data.dataset import test_dataset


def test_full_pipeline():
    """测试完整的数据处理到模型预测流程"""
    print("=== 测试完整流程 ===")
    
    # 1. 构建分子图
    print("1. 测试分子图构建...")
    builder = MolecularGraphBuilder(cutoff_distance=5.0, num_gaussians=16)
    
    try:
        graph_data = builder.build_molecular_graph(
            "./datasets/pdbbind/pdb_files/1a28/1a28.pdb",
            "./datasets/pdbbind/pdb_files/1a28/1a28_ligand.sdf"
        )
        print("✓ 分子图构建成功")
        
        # 2. 创建模型
        print("2. 创建模型...")
        model = create_model_stage1(
            num_atom_types=builder.num_atom_types,
            num_gaussians=builder.num_gaussians
        )
        print("✓ 模型创建成功")
        
        # 3. 准备输入数据
        print("3. 准备输入数据...")
        batch_data = {
            'atom_features': graph_data['atom_features'].unsqueeze(0),  # [1, N, num_atom_types]
            'edge_index': graph_data['edge_index'],                     # [2, E]
            'edge_features': graph_data['edge_features'].unsqueeze(0),  # [1, E, num_gaussians]
            'edge_adjacency': graph_data['edge_adjacency']              # [E, E]
        }
        
        # 4. 前向传播
        print("4. 测试前向传播...")
        model.eval()
        with torch.no_grad():
            predictions = model(batch_data)
        
        print(f"✓ 预测成功！预测值: {predictions.item():.4f}")
        print(f"预测形状: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("PhysESA 阶段一 - 快速测试")
    print("=" * 50)
    
    # 检查数据文件
    test_files = [
        "./datasets/pdbbind/pdb_files/1a28/1a28.pdb",
        "./datasets/pdbbind/pdb_files/1a28/1a28_ligand.sdf"
    ]
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"✗ 测试文件不存在: {file_path}")
            print("请确保数据文件存在后再运行测试")
            return
        else:
            print(f"✓ 找到测试文件: {file_path}")
    
    print()
    
    # 运行各个组件测试
    tests = [
        ("分子图构建器", test_molecular_graph),
        ("ESA组件", test_esa_components),
        ("PhysESA模型", test_model),
        ("完整流程", test_full_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"✓ {test_name} 测试通过")
        except Exception as e:
            print(f"✗ {test_name} 测试失败: {e}")
            results.append((test_name, False))
    
    # 总结
    print(f"\n{'='*20} 测试总结 {'='*20}")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！阶段一模型准备就绪")
        print("\n下一步可以运行:")
        print("python train_stage1.py --num_samples 10 --num_epochs 5")
    else:
        print("❌ 部分测试失败，请检查错误信息")


if __name__ == "__main__":
    main()
