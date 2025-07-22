"""
阶段一：基础框架验证 - 分子图构建器（改进版本）
将氢原子作为重原子的特征，而不是独立节点
只包含重原子作为节点，氢原子数量作为原子特征
"""

import numpy as np
import pandas as pd
import torch
from Bio import PDB
from rdkit import Chem
from torch_geometric.data import Data
import json
import os
from typing import Tuple, Dict, List, Optional

class MolecularGraphBuilder:
    """改进的分子图构建器 - 阶段一版本（重原子+氢原子特征）"""
    
    def __init__(self, cutoff_radius: float = 5.0, num_gaussians: int = 16):
        """
        Args:
            cutoff_radius: 原子间相互作用的截断半径 (Å)
            num_gaussians: 高斯基函数的数量
        """
        self.cutoff_radius = cutoff_radius
        self.num_gaussians = num_gaussians
        
        # 重原子特征维度：原子类型(10) + 氢原子数(1) + 重原子邻居数(1) = 12
        self.atom_feature_dim = 12
        
        # 高斯基函数参数：从0到cutoff_radius均匀分布
        self.gaussian_centers = torch.linspace(0, cutoff_radius, num_gaussians)
        self.gaussian_width = (cutoff_radius / num_gaussians) * 0.5  # β参数
        
    def parse_pdb_file(self, pdb_file: str) -> pd.DataFrame:
        """解析PDB文件，提取重原子信息"""
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('complex', pdb_file)
        
        atoms_data = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        element = atom.element.strip() if atom.element else 'C'
                        
                        # 只保留重原子（非氢原子）
                        if element != 'H':
                            coord = atom.get_coord()
                            atoms_data.append({
                                'element': element,
                                'x': coord[0],
                                'y': coord[1],
                                'z': coord[2],
                                'atom_name': atom.get_name(),
                                'residue_name': residue.get_resname(),
                                'chain_id': chain.id
                            })
        
        return pd.DataFrame(atoms_data)
    
    def parse_sdf_ligand(self, sdf_file: str) -> pd.DataFrame:
        """解析SDF文件，提取配体重原子信息"""
        mol = Chem.MolFromMolFile(sdf_file)
        if mol is None:
            return pd.DataFrame()
        
        conf = mol.GetConformer()
        atoms_data = []
        
        # 统计每个重原子的氢原子数量和重原子邻居数
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != 'H':  # 只保留重原子
                pos = conf.GetAtomPosition(atom.GetIdx())
                
                # 统计氢原子数量
                hydrogen_count = atom.GetTotalNumHs()
                
                # 统计重原子邻居数
                heavy_neighbors = len([n for n in atom.GetNeighbors() if n.GetSymbol() != 'H'])
                
                atoms_data.append({
                    'element': atom.GetSymbol(),
                    'x': pos.x,
                    'y': pos.y,
                    'z': pos.z,
                    'hydrogen_count': hydrogen_count,
                    'heavy_neighbors': heavy_neighbors,
                    'formal_charge': atom.GetFormalCharge(),
                    'atom_idx': atom.GetIdx()
                })
        
        return pd.DataFrame(atoms_data)
    
    def get_atom_type_onehot(self, symbol: str) -> np.ndarray:
        """原子类型独热编码"""
        # 常见重原子元素
        elements = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Other']
        
        if symbol in elements:
            idx = elements.index(symbol)
        else:
            idx = elements.index('Other')
        
        onehot = np.zeros(len(elements))
        onehot[idx] = 1.0
        return onehot
    
    def get_atom_features(self, protein_df: pd.DataFrame, ligand_df: pd.DataFrame) -> torch.Tensor:
        """
        提取原子特征（重原子特征 + 氢原子数量）
        
        Returns:
            shape [num_atoms, atom_feature_dim] 的特征矩阵
        """
        all_features = []
        
        # 处理蛋白质原子
        for _, row in protein_df.iterrows():
            # 原子类型独热编码 (10维)
            atom_type = self.get_atom_type_onehot(row['element'])
            
            # 氢原子数量（对PDB文件，我们无法直接获得，设为0）
            hydrogen_count = 0.0
            
            # 重原子邻居数（对PDB文件，设为合理默认值）
            heavy_neighbors = self._estimate_heavy_neighbors(row['element'])
            
            # 拼接特征
            features = np.concatenate([
                atom_type,           # 10维
                [hydrogen_count],    # 1维
                [heavy_neighbors]    # 1维
            ])
            all_features.append(features)
        
        # 处理配体原子
        for _, row in ligand_df.iterrows():
            # 原子类型独热编码 (10维)
            atom_type = self.get_atom_type_onehot(row['element'])
            
            # 氢原子数量 (1维)
            hydrogen_count = float(row.get('hydrogen_count', 0))
            
            # 重原子邻居数 (1维)
            heavy_neighbors = float(row.get('heavy_neighbors', 0))
            
            # 拼接特征
            features = np.concatenate([
                atom_type,           # 10维
                [hydrogen_count],    # 1维
                [heavy_neighbors]    # 1维
            ])
            all_features.append(features)
        
        return torch.tensor(np.array(all_features), dtype=torch.float32)
    
    def _estimate_heavy_neighbors(self, element: str) -> float:
        """为PDB文件中的原子估计重原子邻居数"""
        # 基于化学知识的粗略估计
        estimates = {
            'C': 2.0,  # 平均值
            'N': 1.5,
            'O': 1.0,
            'S': 2.0,
            'P': 3.0
        }
        return estimates.get(element, 1.0)
    
    def gaussian_basis_functions(self, distances: torch.Tensor) -> torch.Tensor:
        """
        使用高斯基函数扩展距离特征
        
        Args:
            distances: shape [num_edges] 的距离张量
            
        Returns:
            shape [num_edges, num_gaussians] 的扩展特征
        """
        distances = distances.unsqueeze(-1)  # [num_edges, 1]
        centers = self.gaussian_centers.to(distances.device).unsqueeze(0)  # [1, num_gaussians]
        
        # 高斯基函数: exp(-(d - μ)^2 / β^2)
        gaussian_features = torch.exp(
            -((distances - centers) ** 2) / (self.gaussian_width ** 2)
        )
        
        return gaussian_features
    
    def build_graph(self, complex_id: str, pdb_file: str, sdf_file: str) -> Data:
        """
        构建完整的分子图（只包含重原子）
        
        Args:
            complex_id: 复合物ID
            pdb_file: 蛋白质PDB文件路径
            sdf_file: 配体SDF文件路径
            
        Returns:
            PyTorch Geometric Data对象
        """
        # 解析文件
        protein_df = self.parse_pdb_file(pdb_file)
        ligand_df = self.parse_sdf_ligand(sdf_file)
        
        if protein_df.empty:
            raise ValueError(f"无法解析蛋白质文件: {pdb_file}")
        if ligand_df.empty:
            raise ValueError(f"无法解析配体文件: {sdf_file}")
        
        # 合并原子信息
        all_atoms_protein = protein_df[['element', 'x', 'y', 'z']].copy()
        all_atoms_ligand = ligand_df[['element', 'x', 'y', 'z']].copy()
        all_atoms_df = pd.concat([all_atoms_protein, all_atoms_ligand], ignore_index=True)
        all_atoms_df['atom_id'] = range(len(all_atoms_df))
        
        # 提取原子坐标和特征
        positions = torch.tensor(
            all_atoms_df[['x', 'y', 'z']].values, 
            dtype=torch.float32
        )
        atom_features = self.get_atom_features(protein_df, ligand_df)
        
        # 计算原子间距离矩阵
        num_atoms = len(all_atoms_df)
        edge_indices = []
        edge_features = []
        
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):  # 只考虑上三角，无向图
                pos_i = positions[i]
                pos_j = positions[j]
                distance = torch.norm(pos_j - pos_i)
                
                if distance <= self.cutoff_radius:
                    # 添加双向边 (i->j 和 j->i)
                    edge_indices.extend([[i, j], [j, i]])
                    
                    # 使用高斯基函数扩展距离
                    distance_features = self.gaussian_basis_functions(distance.unsqueeze(0))[0]
                    
                    # 对于无向图，两个方向的边特征相同
                    edge_features.extend([distance_features, distance_features])
        
        if not edge_indices:
            raise ValueError(f"复合物 {complex_id} 没有找到任何相互作用边")
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()  # [2, num_edges]
        edge_attr = torch.stack(edge_features)  # [num_edges, num_gaussians]
        
        # 构建PyTorch Geometric Data对象
        data = Data(
            x=atom_features,           # [num_atoms, atom_feature_dim] 
            edge_index=edge_index,     # [2, num_edges]
            edge_attr=edge_attr,       # [num_edges, num_gaussians]
            pos=positions,             # [num_atoms, 3]
            complex_id=complex_id,
            num_protein_atoms=len(protein_df),
            num_ligand_atoms=len(ligand_df)
        )
        
        return data
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """返回特征维度信息"""
        return {
            'node_dim': self.atom_feature_dim,  # 12维: 原子类型(10) + 氢原子数(1) + 重原子邻居数(1)
            'edge_dim': self.num_gaussians,     # 16维: 高斯基函数扩展
            'pos_dim': 3
        }

# 辅助函数
def load_pdbbind_metadata(metadata_path: str) -> Dict:
    """加载PDBbind数据集的元数据"""
    metadata = {}
    
    try:
        # 亲和力数据
        affinity_file = os.path.join(metadata_path, 'affinities.json')
        if os.path.exists(affinity_file):
            with open(affinity_file, 'r') as f:
                metadata['affinities'] = json.load(f)
        
        # SMILES数据（可选）
        smiles_file = os.path.join(metadata_path, 'lig_smiles.json')
        if os.path.exists(smiles_file):
            with open(smiles_file, 'r') as f:
                metadata['smiles'] = json.load(f)
                
    except Exception as e:
        print(f"加载元数据时出现警告: {e}")
    
    return metadata

if __name__ == "__main__":
    # 测试代码
    builder = MolecularGraphBuilder(cutoff_radius=5.0, num_gaussians=16)
    
    # 测试1a28复合物
    complex_id = "1a28"
    pdb_file = f"./datasets/pdbbind/pdb_files/{complex_id}/{complex_id}.pdb"
    sdf_file = f"./datasets/pdbbind/pdb_files/{complex_id}/{complex_id}_ligand.sdf"
    
    if os.path.exists(pdb_file) and os.path.exists(sdf_file):
        try:
            data = builder.build_graph(complex_id, pdb_file, sdf_file)
            print(f"成功构建图: {complex_id}")
            print(f"重原子数: {data.x.shape[0]}")
            print(f"边数: {data.edge_index.shape[1]}")
            print(f"原子特征维度: {data.x.shape[1]}")
            print(f"边特征维度: {data.edge_attr.shape[1]}")
            print(f"蛋白质重原子数: {data.num_protein_atoms}")
            print(f"配体重原子数: {data.num_ligand_atoms}")
            
            # 检查特征维度
            dims = builder.get_feature_dimensions()
            print(f"特征维度: {dims}")
            
            # 检查一些配体原子的特征
            ligand_start_idx = data.num_protein_atoms
            if data.num_ligand_atoms > 0:
                print("\n配体原子特征示例:")
                for i in range(min(5, data.num_ligand_atoms)):
                    idx = ligand_start_idx + i
                    features = data.x[idx]
                    # 解析特征
                    atom_type = torch.argmax(features[:10]).item()
                    hydrogen_count = features[10].item()
                    heavy_neighbors = features[11].item()
                    
                    atom_names = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Other']
                    print(f"  原子{i}: 类型={atom_names[atom_type]}, "
                          f"氢原子数={hydrogen_count:.0f}, "
                          f"重原子邻居数={heavy_neighbors:.0f}")
            
        except Exception as e:
            print(f"构建图时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"找不到文件: {pdb_file} 或 {sdf_file}")
