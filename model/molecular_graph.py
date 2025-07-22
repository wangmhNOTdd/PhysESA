"""
阶段一：基础框架验证 - 分子图构建器
简化版本：只使用原子类型（独热编码）和高斯基函数扩展的距离
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
    """简化的分子图构建器 - 阶段一版本"""
    
    def __init__(self, cutoff_radius: float = 5.0, num_gaussians: int = 16):
        """
        Args:
            cutoff_radius: 原子间相互作用的截断半径 (Å)
            num_gaussians: 高斯基函数的数量
        """
        self.cutoff_radius = cutoff_radius
        self.num_gaussians = num_gaussians
        
        # 常见生物分子元素映射
        self.atom_types = {
            'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'P': 5, 
            'F': 6, 'Cl': 7, 'Br': 8, 'I': 9
        }
        self.num_atom_types = len(self.atom_types)
        
        # 高斯基函数参数：从0到cutoff_radius均匀分布
        self.gaussian_centers = torch.linspace(0, cutoff_radius, num_gaussians)
        self.gaussian_width = (cutoff_radius / num_gaussians) * 0.5  # β参数
        
    def parse_pdb_file(self, pdb_file: str) -> pd.DataFrame:
        """解析PDB文件，提取原子信息"""
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('complex', pdb_file)
        
        atoms_data = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        coord = atom.get_coord()
                        element = atom.element.strip() if atom.element else 'C'
                        
                        atoms_data.append({
                            'atom_id': len(atoms_data),
                            'element': element,
                            'x': coord[0],
                            'y': coord[1], 
                            'z': coord[2],
                            'chain_id': chain.id,
                            'residue_name': residue.get_resname(),
                            'atom_type': 'protein'
                        })
        
        return pd.DataFrame(atoms_data)
    
    def parse_sdf_ligand(self, sdf_file: str) -> pd.DataFrame:
        """解析SDF文件，提取配体原子信息"""
        mol = Chem.MolFromMolFile(sdf_file)
        if mol is None:
            return pd.DataFrame()
        
        conf = mol.GetConformer()
        atoms_data = []
        
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            element = atom.GetSymbol()
            
            atoms_data.append({
                'atom_id': atom.GetIdx(),
                'element': element,
                'x': pos.x,
                'y': pos.y,
                'z': pos.z,
                'chain_id': 'L',  # 配体链
                'residue_name': 'LIG',
                'atom_type': 'ligand'
            })
            
        return pd.DataFrame(atoms_data)
    
    def get_atom_features(self, atoms_df: pd.DataFrame) -> torch.Tensor:
        """
        构建原子特征矩阵（阶段一：只使用原子类型的独热编码）
        
        Args:
            atoms_df: 包含原子信息的DataFrame
            
        Returns:
            shape: [num_atoms, num_atom_types] 的特征矩阵
        """
        num_atoms = len(atoms_df)
        atom_features = torch.zeros(num_atoms, self.num_atom_types)
        
        for idx, row in atoms_df.iterrows():
            element = row['element']
            if element in self.atom_types:
                atom_type_idx = self.atom_types[element]
                atom_features[idx, atom_type_idx] = 1.0
            else:
                # 未知元素，使用碳原子作为默认
                atom_features[idx, self.atom_types['C']] = 1.0
                
        return atom_features
    
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
        构建完整的分子图
        
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
        
        # 合并原子信息，重新分配atom_id
        all_atoms_df = pd.concat([protein_df, ligand_df], ignore_index=True)
        all_atoms_df['atom_id'] = range(len(all_atoms_df))
        
        # 提取原子坐标和特征
        positions = torch.tensor(
            all_atoms_df[['x', 'y', 'z']].values, 
            dtype=torch.float32
        )
        atom_features = self.get_atom_features(all_atoms_df)
        
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
            x=atom_features,           # [num_atoms, num_atom_types] 
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
            'node_dim': self.num_atom_types,
            'edge_dim': self.num_gaussians,
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
            print(f"原子数: {data.x.shape[0]}")
            print(f"边数: {data.edge_index.shape[1]}")
            print(f"原子特征维度: {data.x.shape[1]}")
            print(f"边特征维度: {data.edge_attr.shape[1]}")
            print(f"蛋白质原子数: {data.num_protein_atoms}")
            print(f"配体原子数: {data.num_ligand_atoms}")
            
            # 检查特征维度
            dims = builder.get_feature_dimensions()
            print(f"特征维度: {dims}")
            
        except Exception as e:
            print(f"构建图时出错: {e}")
    else:
        print(f"找不到文件: {pdb_file} 或 {sdf_file}")
