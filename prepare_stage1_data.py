"""
阶段一数据预处理脚本
简化特征，生成可训练的数据集文件
"""

import os
import pickle
import torch
import json
import numpy as np
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import argparse

from molecular_graph import MolecularGraphBuilder, load_pdbbind_metadata


class Stage1GraphBuilder(MolecularGraphBuilder):
    """阶段一简化版图构建器"""
    
    def __init__(self, cutoff_radius: float = 5.0, num_gaussians: int = 16, use_knn: bool = True, k: int = 16):
        super().__init__(cutoff_radius, num_gaussians, use_knn, k)
        # 阶段一：简化特征维度
        # 原子特征：只使用原子类型独热编码 (10维)
        self.atom_feature_dim = 10
        
    def get_atom_features(self, protein_df, ligand_df) -> torch.Tensor:
        """
        阶段一简化版：只使用原子类型特征
        注意：protein_df 和 ligand_df 可能是界面过滤后的数据
        """
        all_features = []
        
        # 处理蛋白质原子
        if not protein_df.empty:
            for _, row in protein_df.iterrows():
                # 只使用原子类型独热编码 (10维)
                atom_type = self.get_atom_type_onehot(row['element'])
                all_features.append(atom_type)
        
        # 处理配体原子
        if not ligand_df.empty:
            for _, row in ligand_df.iterrows():
                # 只使用原子类型独热编码 (10维)
                atom_type = self.get_atom_type_onehot(row['element'])
                all_features.append(atom_type)
        
        if len(all_features) == 0:
            raise ValueError("没有找到任何原子特征")
        
        return torch.tensor(np.array(all_features), dtype=torch.float32)
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """返回阶段一的特征维度信息"""
        return {
            'node_dim': self.atom_feature_dim,  # 10维: 原子类型独热编码
            'edge_dim': self.num_gaussians,     # 16维: 高斯基函数扩展距离
            'pos_dim': 3
        }


def prepare_split_data(
    data_root: str,
    split_type: str = "scaffold_split",
    max_samples_per_split: Optional[int] = None
) -> Tuple[List[str], List[str], List[str]]:
    """
    准备数据分割
    
    Args:
        data_root: 数据根目录
        split_type: 分割类型 ("scaffold_split", "identity30_split", "identity60_split")
        max_samples_per_split: 每个分割的最大样本数（用于调试）
    
    Returns:
        (train_ids, val_ids, test_ids)
    """
    metadata_path = os.path.join(data_root, 'metadata')
    split_file = os.path.join(metadata_path, f'{split_type}.json')
    
    if os.path.exists(split_file):
        print(f"使用预定义分割: {split_file}")
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        train_ids = split_data.get('train', [])
        val_ids = split_data.get('val', split_data.get('valid', []))  # 兼容不同命名
        test_ids = split_data.get('test', [])
        
        # 如果预定义分割中没有验证集，从训练集中分出一部分
        if len(val_ids) == 0 and len(train_ids) > 0:
            print("预定义分割中没有验证集，从训练集分出10%作为验证集")
            n_val = max(1, len(train_ids) // 10)  # 至少1个样本
            val_ids = train_ids[-n_val:]
            train_ids = train_ids[:-n_val]
        
        # 限制样本数量（用于调试）
        if max_samples_per_split:
            train_ids = train_ids[:max_samples_per_split]
            val_ids = val_ids[:max(1, max_samples_per_split//10)]  # 验证集至少1个
            test_ids = test_ids[:max(1, max_samples_per_split//10)]  # 测试集至少1个
            
    else:
        print("使用简单的8:1:1分割")
        # 扫描所有可用的复合物
        pdb_files_dir = os.path.join(data_root, 'pdb_files')
        all_ids = [d for d in os.listdir(pdb_files_dir) 
                  if os.path.isdir(os.path.join(pdb_files_dir, d))]
        
        if max_samples_per_split:
            all_ids = all_ids[:int(max_samples_per_split * 1.25)]  # 稍微多一点以保证足够的样本
            
        # 简单分割
        n_total = len(all_ids)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        
        train_ids = all_ids[:n_train]
        val_ids = all_ids[n_train:n_train + n_val]
        test_ids = all_ids[n_train + n_val:]
    
    print(f"数据分割统计:")
    print(f"  训练集: {len(train_ids)}")
    print(f"  验证集: {len(val_ids)}")
    print(f"  测试集: {len(test_ids)}")
    
    return train_ids, val_ids, test_ids


def process_complex_list(
    complex_ids: List[str],
    data_root: str,
    graph_builder: Stage1GraphBuilder,
    metadata: Dict,
    split_name: str
) -> List[Tuple[Data, float, str]]:
    """
    处理复合物列表，返回有效的图数据
    
    Returns:
        List of (graph_data, affinity, complex_id)
    """
    processed_data = []
    
    print(f"处理{split_name}数据...")
    for complex_id in tqdm(complex_ids, desc=f"处理{split_name}"):
        # 构建文件路径
        pdb_file = os.path.join(data_root, 'pdb_files', complex_id, f'{complex_id}.pdb')
        sdf_file = os.path.join(data_root, 'pdb_files', complex_id, f'{complex_id}_ligand.sdf')
        
        # 检查文件是否存在
        if not (os.path.exists(pdb_file) and os.path.exists(sdf_file)):
            print(f"跳过 {complex_id}: 文件缺失")
            continue
            
        # 获取亲和力标签
        if 'affinities' in metadata and complex_id in metadata['affinities']:
            affinity = float(metadata['affinities'][complex_id])
        else:
            print(f"跳过 {complex_id}: 缺少亲和力数据")
            continue
            
        # 尝试构建图
        try:
            graph_data = graph_builder.build_graph(complex_id, pdb_file, sdf_file)
            processed_data.append((graph_data, affinity, complex_id))
            
        except Exception as e:
            print(f"跳过 {complex_id}: 图构建失败 - {e}")
            continue
    
    print(f"{split_name}有效样本数: {len(processed_data)}")
    return processed_data


def prepare_for_esa_training(graph_data: Data) -> Dict:
    """
    将图数据转换为ESA训练所需的格式
    """
    # 计算边的数量，用于后续的批处理
    num_edges = graph_data.edge_index.shape[1]
    
    # 为ESA准备数据：每条边关联其两端的节点特征
    edge_index = graph_data.edge_index
    x = graph_data.x  # 节点特征
    edge_attr = graph_data.edge_attr  # 边特征
    
    # ESA需要的格式：将每条边表示为 [src_node_feat, tgt_node_feat, edge_feat]
    source_features = x[edge_index[0]]  # [num_edges, node_dim]
    target_features = x[edge_index[1]]  # [num_edges, node_dim]
    
    # 拼接边表示：[src_feat, tgt_feat, edge_feat]
    if edge_attr is not None:
        edge_representations = torch.cat([source_features, target_features, edge_attr], dim=1)
    else:
        edge_representations = torch.cat([source_features, target_features], dim=1)
    
    return {
        'edge_representations': edge_representations,  # [num_edges, feature_dim]
        'edge_index': edge_index,  # [2, num_edges] 
        'num_edges': num_edges,
        'num_nodes': graph_data.x.shape[0],
        'node_features': x,  # 保留原始节点特征以备用
        'edge_features': edge_attr,  # 保留原始边特征
        'pos': graph_data.pos,  # 原子坐标
        'complex_id': graph_data.complex_id,
        'num_protein_atoms': graph_data.num_protein_atoms,
        'num_ligand_atoms': graph_data.num_ligand_atoms
    }


def test_single_sample(
    complex_id: str,
    data_root: str,
    graph_builder: Stage1GraphBuilder,
    metadata: Dict,
    save_output: bool = False,
    output_dir: str = './experiments/stage1'
) -> None:
    """
    测试单个样本的处理过程
    """
    print(f"\n=== 测试样本: {complex_id} ===")
    
    # 构建文件路径
    pdb_file = os.path.join(data_root, 'pdb_files', complex_id, f'{complex_id}.pdb')
    sdf_file = os.path.join(data_root, 'pdb_files', complex_id, f'{complex_id}_ligand.sdf')
    
    # 检查文件
    print(f"PDB文件: {pdb_file}")
    print(f"存在: {os.path.exists(pdb_file)}")
    print(f"SDF文件: {sdf_file}")
    print(f"存在: {os.path.exists(sdf_file)}")
    
    if not (os.path.exists(pdb_file) and os.path.exists(sdf_file)):
        print("❌ 文件缺失，无法处理")
        return
    
    # 获取亲和力
    if 'affinities' in metadata and complex_id in metadata['affinities']:
        affinity = float(metadata['affinities'][complex_id])
        print(f"亲和力: {affinity}")
    else:
        print("❌ 缺少亲和力数据")
        return
    
    try:
        # 构建图
        print("\n--- 构建分子图 ---")
        graph_data = graph_builder.build_graph(complex_id, pdb_file, sdf_file)
        
        print(f"✅ 图构建成功!")
        print(f"节点数: {graph_data.x.shape[0]}")
        print(f"边数: {graph_data.edge_index.shape[1]}")
        print(f"节点特征维度: {graph_data.x.shape[1]}")
        print(f"边特征维度: {graph_data.edge_attr.shape[1] if graph_data.edge_attr is not None else 0}")
        print(f"蛋白质原子数: {graph_data.num_protein_atoms}")
        print(f"配体原子数: {graph_data.num_ligand_atoms}")
        
        # 转换为ESA格式
        print("\n--- 转换为ESA格式 ---")
        esa_sample = prepare_for_esa_training(graph_data)
        esa_sample['affinity'] = affinity
        
        print(f"边表示维度: {esa_sample['edge_representations'].shape}")
        print(f"边表示范围: [{esa_sample['edge_representations'].min():.3f}, {esa_sample['edge_representations'].max():.3f}]")
        
        # 保存测试输出
        if save_output:
            os.makedirs(output_dir, exist_ok=True)
            test_output = {
                'complex_id': complex_id,
                'graph_data': graph_data,
                'esa_sample': esa_sample,
                'files': {'pdb': pdb_file, 'sdf': sdf_file},
                'affinity': affinity
            }
            
            output_file = os.path.join(output_dir, f'test_sample_{complex_id}.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump(test_output, f)
            print(f"测试输出已保存到: {output_file}")
        
        print("✅ 样本处理成功!")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='阶段一数据预处理')
    parser.add_argument('--data_root', type=str, default='./datasets/pdbbind', 
                       help='PDBbind数据集根目录')
    parser.add_argument('--output_dir', type=str, default='./experiments/stage1',
                       help='输出目录')
    parser.add_argument('--split_type', type=str, default='scaffold_split',
                       choices=['scaffold_split', 'identity30_split', 'identity60_split'],
                       help='数据分割类型')
    parser.add_argument('--cutoff_radius', type=float, default=5.0,
                       help='原子相互作用截断半径（仅在使用radius模式时生效）')
    parser.add_argument('--num_gaussians', type=int, default=16,
                       help='高斯基函数数量')
    parser.add_argument('--use_knn', action='store_true', default=True,
                       help='使用KNN连边方式（默认启用）')
    parser.add_argument('--use_radius', action='store_true',
                       help='使用截断半径连边方式（覆盖--use_knn）')
    parser.add_argument('--k', type=int, default=16,
                       help='KNN中的k值（每个原子连接最近的k个原子）')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='每个分割的最大样本数（用于调试）')
    parser.add_argument('--test_run', action='store_true',
                       help='测试运行模式，只处理少量样本')
    parser.add_argument('--test_sample', type=str, default=None,
                       help='测试单个样本（提供复合物ID，如1a4k）')
    parser.add_argument('--save_test_output', action='store_true',
                       help='保存测试样本的详细输出信息')
    
    args = parser.parse_args()
    
    # 测试模式
    if args.test_run:
        args.max_samples = 50
        print("*** 测试运行模式：仅处理50个样本 ***")
    
    # 确定连边方式
    use_knn = args.use_knn and not args.use_radius  # 如果指定了use_radius，就不用KNN
    
    print(f"连边方式: {'KNN (k={})'.format(args.k) if use_knn else '截断半径 (r={}Å)'.format(args.cutoff_radius)}")
    
    # 检查数据目录
    if not os.path.exists(args.data_root):
        raise ValueError(f"数据目录不存在: {args.data_root}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化图构建器
    graph_builder = Stage1GraphBuilder(
        cutoff_radius=args.cutoff_radius,
        num_gaussians=args.num_gaussians,
        use_knn=use_knn,
        k=args.k
    )
    
    # 加载元数据
    metadata_path = os.path.join(args.data_root, 'metadata')
    metadata = load_pdbbind_metadata(metadata_path)
    
    # 如果是单样本测试模式
    if args.test_sample:
        test_single_sample(
            args.test_sample,
            args.data_root,
            graph_builder,
            metadata,
            args.save_test_output,
            args.output_dir
        )
        return
    
    # 准备数据分割
    train_ids, val_ids, test_ids = prepare_split_data(
        args.data_root, 
        args.split_type,
        args.max_samples
    )
    
    # 处理各个数据集
    datasets = {
        'train': (train_ids, 'train'),
        'valid': (val_ids, 'valid'),
        'test': (test_ids, 'test')
    }
    
    feature_stats = {}
    
    for split_name, (complex_ids, filename) in datasets.items():
        if len(complex_ids) == 0:
            print(f"警告: {split_name}集为空，跳过")
            continue
            
        # 处理数据
        processed_data = process_complex_list(
            complex_ids, args.data_root, graph_builder, metadata, split_name
        )
        
        if len(processed_data) == 0:
            print(f"错误: {split_name}集没有有效数据")
            continue
        
        # 转换为ESA格式并添加统计信息
        esa_data = []
        edge_counts = []
        node_counts = []
        affinities = []
        
        for graph_data, affinity, complex_id in processed_data:
            esa_sample = prepare_for_esa_training(graph_data)
            esa_sample['affinity'] = affinity
            esa_data.append(esa_sample)
            
            # 收集统计信息
            edge_counts.append(esa_sample['num_edges'])
            node_counts.append(esa_sample['num_nodes'])
            affinities.append(affinity)
        
        # 保存数据
        output_file = os.path.join(args.output_dir, f'{filename}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(esa_data, f)
        
        print(f"{split_name}数据已保存到: {output_file}")
        
        # 统计信息
        feature_stats[split_name] = {
            'num_samples': len(esa_data),
            'edge_count_stats': {
                'mean': np.mean(edge_counts),
                'std': np.std(edge_counts),
                'min': np.min(edge_counts),
                'max': np.max(edge_counts)
            },
            'node_count_stats': {
                'mean': np.mean(node_counts),
                'std': np.std(node_counts),
                'min': np.min(node_counts),
                'max': np.max(node_counts)
            },
            'affinity_stats': {
                'mean': np.mean(affinities),
                'std': np.std(affinities),
                'min': np.min(affinities),
                'max': np.max(affinities)
            }
        }
    
    # 保存特征维度信息和统计信息
    feature_dims = graph_builder.get_feature_dimensions()
    
    metadata_info = {
        'feature_dimensions': feature_dims,
        'dataset_stats': feature_stats,
        'cutoff_radius': args.cutoff_radius,
        'num_gaussians': args.num_gaussians,
        'split_type': args.split_type,
        'edge_connection': {
            'method': 'knn' if use_knn else 'radius',
            'k': args.k if use_knn else None,
            'radius': args.cutoff_radius if not use_knn else None
        },
        'data_format': 'esa_edge_representations'
    }
    
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata_info, f, indent=2, default=str)
    
    print("\n=== 数据预处理完成 ===")
    print(f"特征维度: {feature_dims}")
    print(f"输出目录: {args.output_dir}")
    
    print("\n=== 数据集统计 ===")
    for split_name, stats in feature_stats.items():
        print(f"\n{split_name.upper()}:")
        print(f"  样本数: {stats['num_samples']}")
        print(f"  边数 - 均值: {stats['edge_count_stats']['mean']:.1f}, "
              f"范围: [{stats['edge_count_stats']['min']}, {stats['edge_count_stats']['max']}]")
        print(f"  节点数 - 均值: {stats['node_count_stats']['mean']:.1f}, "
              f"范围: [{stats['node_count_stats']['min']}, {stats['node_count_stats']['max']}]")
        print(f"  亲和力 - 均值: {stats['affinity_stats']['mean']:.2f}, "
              f"标准差: {stats['affinity_stats']['std']:.2f}")


if __name__ == "__main__":
    main()
