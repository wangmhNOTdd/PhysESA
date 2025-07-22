"""
阶段一：PDBbind数据集加载器
简化版本，支持批量加载和处理
"""

import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from typing import List, Dict, Optional, Tuple
from molecular_graph import MolecularGraphBuilder, load_pdbbind_metadata

class PDBBindDataset(Dataset):
    """PDBbind数据集 - 阶段一版本"""
    
    def __init__(
        self,
        data_root: str,
        split_file: Optional[str] = None,
        complex_ids: Optional[List[str]] = None,
        cutoff_radius: float = 5.0,
        num_gaussians: int = 16,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_root: 数据集根目录
            split_file: 分割文件路径（可选）
            complex_ids: 指定的复合物ID列表（可选）
            cutoff_radius: 原子相互作用截断半径
            num_gaussians: 高斯基函数数量
            max_samples: 最大样本数量（用于调试）
        """
        self.data_root = data_root
        self.cutoff_radius = cutoff_radius
        self.num_gaussians = num_gaussians
        
        # 初始化图构建器
        self.graph_builder = MolecularGraphBuilder(cutoff_radius, num_gaussians)
        
        # 加载元数据
        metadata_path = os.path.join(data_root, 'metadata')
        self.metadata = load_pdbbind_metadata(metadata_path)
        
        # 确定要处理的复合物ID列表
        self.complex_ids = self._get_complex_ids(split_file, complex_ids, max_samples)
        
        # 过滤有效的复合物（存在PDB和SDF文件）
        self.valid_complex_ids = self._filter_valid_complexes()
        
        print(f"数据集初始化完成:")
        print(f"  - 总复合物数: {len(self.complex_ids)}")
        print(f"  - 有效复合物数: {len(self.valid_complex_ids)}")
        print(f"  - 截断半径: {cutoff_radius}Å")
        print(f"  - 高斯基函数数量: {num_gaussians}")
        
    def _get_complex_ids(
        self, 
        split_file: Optional[str], 
        complex_ids: Optional[List[str]],
        max_samples: Optional[int]
    ) -> List[str]:
        """获取要处理的复合物ID列表"""
        
        if complex_ids is not None:
            ids = complex_ids
        elif split_file is not None and os.path.exists(split_file):
            # 从分割文件加载
            with open(split_file, 'r') as f:
                split_data = json.load(f)
                ids = split_data.get('train', []) + split_data.get('val', []) + split_data.get('test', [])
        else:
            # 扫描PDB文件目录
            pdb_files_dir = os.path.join(self.data_root, 'pdb_files')
            if os.path.exists(pdb_files_dir):
                ids = [d for d in os.listdir(pdb_files_dir) 
                      if os.path.isdir(os.path.join(pdb_files_dir, d))]
            else:
                raise ValueError(f"找不到PDB文件目录: {pdb_files_dir}")
        
        # 限制样本数量（用于调试）
        if max_samples is not None:
            ids = ids[:max_samples]
            
        return sorted(ids)
    
    def _filter_valid_complexes(self) -> List[str]:
        """过滤存在PDB和SDF文件的有效复合物，并测试是否能够构建图"""
        valid_ids = []
        
        for complex_id in self.complex_ids:
            pdb_file = self._get_pdb_file_path(complex_id)
            sdf_file = self._get_sdf_file_path(complex_id)
            
            # 检查文件是否存在
            if not (os.path.exists(pdb_file) and os.path.exists(sdf_file)):
                print(f"跳过复合物 {complex_id}: 文件缺失")
                continue
            
            # 测试是否能够构建分子图
            try:
                # 快速测试构建图（不保存结果）
                self.graph_builder.build_graph(complex_id, pdb_file, sdf_file)
                valid_ids.append(complex_id)
            except Exception as e:
                print(f"跳过复合物 {complex_id}: 图构建失败 - {e}")
                continue
                
        return valid_ids
    
    def _get_pdb_file_path(self, complex_id: str) -> str:
        """获取PDB文件路径"""
        return os.path.join(self.data_root, 'pdb_files', complex_id, f'{complex_id}.pdb')
    
    def _get_sdf_file_path(self, complex_id: str) -> str:
        """获取SDF文件路径"""
        return os.path.join(self.data_root, 'pdb_files', complex_id, f'{complex_id}_ligand.sdf')
    
    def _get_affinity(self, complex_id: str) -> float:
        """获取结合亲和力标签"""
        if 'affinities' in self.metadata and complex_id in self.metadata['affinities']:
            return float(self.metadata['affinities'][complex_id])
        else:
            print(f"警告: 复合物 {complex_id} 缺少亲和力数据，使用默认值 5.0")
            return 5.0  # 默认值
    
    def __len__(self) -> int:
        return len(self.valid_complex_ids)
    
    def __getitem__(self, idx: int) -> Tuple[Data, float]:
        """获取单个样本"""
        complex_id = self.valid_complex_ids[idx]
        
        # 获取文件路径
        pdb_file = self._get_pdb_file_path(complex_id)
        sdf_file = self._get_sdf_file_path(complex_id)
        
        # 构建分子图（已经在初始化时验证过，理论上应该成功）
        data = self.graph_builder.build_graph(complex_id, pdb_file, sdf_file)
        
        # 获取标签
        affinity = self._get_affinity(complex_id)
        
        return data, affinity
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """获取特征维度信息"""
        return self.graph_builder.get_feature_dimensions()

def collate_fn(batch: List[Tuple[Data, float]]) -> Tuple[List[Data], torch.Tensor]:
    """
    自定义collate函数，处理不同大小的图
    注意：这里不使用PyTorch Geometric的Batch，而是返回列表
    """
    graphs, affinities = zip(*batch)
    affinity_tensor = torch.tensor(affinities, dtype=torch.float32)
    return list(graphs), affinity_tensor

def create_data_loaders(
    data_root: str,
    train_ids: Optional[List[str]] = None,
    val_ids: Optional[List[str]] = None,
    test_ids: Optional[List[str]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    cutoff_radius: float = 5.0,
    num_gaussians: int = 16,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    """
    
    # 如果没有提供分割，使用简单的8:1:1分割
    if train_ids is None and val_ids is None and test_ids is None:
        # 获取所有可用的复合物ID
        pdb_files_dir = os.path.join(data_root, 'pdb_files')
        all_ids = [d for d in os.listdir(pdb_files_dir) 
                  if os.path.isdir(os.path.join(pdb_files_dir, d))]
        
        if max_samples:
            all_ids = all_ids[:max_samples]
            
        # 简单分割
        n_total = len(all_ids)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        
        train_ids = all_ids[:n_train]
        val_ids = all_ids[n_train:n_train + n_val]
        test_ids = all_ids[n_train + n_val:]
        
        print(f"数据分割:")
        print(f"  总复合物: {n_total}")
        print(f"  训练集: {len(train_ids)}")
        print(f"  验证集: {len(val_ids)}")
        print(f"  测试集: {len(test_ids)}")
    
    # 创建数据集
    train_dataset = PDBBindDataset(
        data_root, complex_ids=train_ids, 
        cutoff_radius=cutoff_radius, num_gaussians=num_gaussians
    )
    
    val_dataset = PDBBindDataset(
        data_root, complex_ids=val_ids,
        cutoff_radius=cutoff_radius, num_gaussians=num_gaussians
    )
    
    test_dataset = PDBBindDataset(
        data_root, complex_ids=test_ids,
        cutoff_radius=cutoff_radius, num_gaussians=num_gaussians
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,  # 阶段一先使用batch_size=1
        shuffle=True,
        num_workers=0,  # 避免多进程问题
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # 测试数据加载器
    data_root = "./datasets/pdbbind"
    
    if os.path.exists(data_root):
        # 创建小规模测试数据加载器
        train_loader, val_loader, test_loader = create_data_loaders(
            data_root=data_root,
            batch_size=1,
            max_samples=10  # 仅测试10个样本
        )
        
        print("数据加载器测试:")
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        print(f"测试集大小: {len(test_loader.dataset)}")
        
        # 测试加载一个批次
        try:
            graphs, affinities = next(iter(train_loader))
            print(f"批次测试成功:")
            print(f"  图数量: {len(graphs)}")
            print(f"  亲和力: {affinities}")
            
            # 检查第一个图的结构
            first_graph = graphs[0]
            print(f"  第一个图: 原子数={first_graph.x.shape[0]}, 边数={first_graph.edge_index.shape[1]}")
            
        except Exception as e:
            print(f"批次加载失败: {e}")
    
    else:
        print(f"数据目录不存在: {data_root}")
