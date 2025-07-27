"""
阶段二数据预处理脚本
使用Stage2GraphBuilder生成包含完整3D物理信息的图数据
"""

import os
import pickle
import json
import torch
import numpy as np
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import argparse
import sys
import traceback

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from molecular_graph import MultiScaleGraphBuilder, load_pdbbind_metadata, prepare_split_data

def process_complex_list(
    complex_ids: List[str],
    data_root: str,
    graph_builder: MultiScaleGraphBuilder,
    metadata: Dict,
    split_name: str
) -> List[Data]:
    """
    处理复合物列表，返回有效的图数据列表
    """
    processed_data = []
    
    print(f"处理 {split_name} 数据...")
    for complex_id in tqdm(complex_ids, desc=f"处理 {split_name}"):
        pdb_file = os.path.join(data_root, 'pdb_files', complex_id, f'{complex_id}.pdb')
        sdf_file = os.path.join(data_root, 'pdb_files', complex_id, f'{complex_id}_ligand.sdf')
        
        if not (os.path.exists(pdb_file) and os.path.exists(sdf_file)):
            # print(f"跳过 {complex_id}: 文件缺失")
            continue
            
        if 'affinities' in metadata and complex_id in metadata['affinities']:
            affinity = float(metadata['affinities'][complex_id])
        else:
            # print(f"跳过 {complex_id}: 缺少亲和力数据")
            continue
            
        try:
            graph_data = graph_builder.build_graph(complex_id, pdb_file, sdf_file)
            if graph_data is not None:
                graph_data.y = torch.tensor([affinity], dtype=torch.float32)
                processed_data.append(graph_data)
        except Exception as e:
            print(f"错误: 跳过 {complex_id}: 图构建失败 - {e}")
            traceback.print_exc() # 打印详细的堆栈跟踪
            continue
    
    print(f"{split_name} 有效样本数: {len(processed_data)}")
    return processed_data

def main():
    parser = argparse.ArgumentParser(description='阶段二数据预处理')
    parser.add_argument('--data_root', type=str, default='./datasets/pdbbind', 
                       help='PDBbind数据集根目录')
    parser.add_argument('--output_dir', type=str, default='./experiments/stage2',
                       help='输出目录')
    parser.add_argument('--split_type', type=str, default='identity60_split',
                       choices=['scaffold_split', 'identity30_split', 'identity60_split'],
                       help='数据分割类型')
    parser.add_argument('--interface_cutoff', type=float, default=8.0,
                       help='定义复合物界面的距离阈值 (Å)')
    parser.add_argument('--cutoff_radius', type=float, default=5.0,
                       help='原子相互作用截断半径（仅在使用radius模式时生效）')
    parser.add_argument('--num_gaussians', type=int, default=16,
                       help='高斯基函数数量')
    parser.add_argument('--use_knn', action='store_true', default=True,
                       help='使用KNN连边方式（默认启用）')
    parser.add_argument('--use_radius', action='store_true',
                       help='使用截断半径连边方式（覆盖--use_knn）')
    parser.add_argument('--k', type=int, default=16,
                       help='KNN中的k值')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='每个分割的最大样本数（用于调试）')
    parser.add_argument('--test_run', action='store_true',
                       help='测试运行模式，只处理少量样本')
    
    args = parser.parse_args()
    
    if args.test_run:
        args.max_samples = 50
        print("*** 测试运行模式：仅处理约50个样本 ***")
    
    use_knn = args.use_knn and not args.use_radius
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    graph_builder = MultiScaleGraphBuilder(
        cutoff_radius=args.cutoff_radius,
        num_gaussians=args.num_gaussians,
        use_knn=use_knn,
        k=args.k,
        interface_cutoff=args.interface_cutoff
    )
    
    metadata_path = os.path.join(args.data_root, 'metadata')
    metadata = load_pdbbind_metadata(metadata_path)
    
    train_ids, val_ids, test_ids = prepare_split_data(
        args.data_root, 
        args.split_type,
        args.max_samples
    )
    
    datasets = {
        'train': (train_ids, 'train.pkl'),
        'valid': (val_ids, 'valid.pkl'),
        'test': (test_ids, 'test.pkl')
    }
    
    for split_name, (complex_ids, filename) in datasets.items():
        if not complex_ids:
            print(f"警告: {split_name} 集为空，跳过")
            continue
            
        processed_data = process_complex_list(
            complex_ids, args.data_root, graph_builder, metadata, split_name
        )
        
        if not processed_data:
            print(f"错误: {split_name} 集没有有效数据")
            continue
        
        output_file = os.path.join(args.output_dir, filename)
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"{split_name} 数据已保存到: {output_file}")

    # 保存元数据
    metadata_info = {
        'feature_dimensions': graph_builder.get_feature_dimensions(),
        'graph_builder_config': {
            'cutoff_radius': args.cutoff_radius,
            'num_gaussians': args.num_gaussians,
            'use_knn': use_knn,
            'k': args.k,
            'interface_cutoff': args.interface_cutoff
        }
    }
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata_info, f, indent=2)
        
    print("\n=== 阶段二数据预处理完成 ===")
    print(f"特征维度: {metadata_info['feature_dimensions']}")
    print(f"输出目录: {args.output_dir}")

if __name__ == "__main__":
    main()