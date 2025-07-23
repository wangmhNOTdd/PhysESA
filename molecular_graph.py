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
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import json
import os
from typing import Tuple, Dict, List, Optional

class MolecularGraphBuilder:
    """改进的分子图构建器 - 阶段一版本（重原子+氢原子特征）"""
    
    def __init__(self, cutoff_radius: float = 5.0, num_gaussians: int = 16, use_knn: bool = False, k: int = 16, interface_cutoff: float = 8.0):
        """
        Args:
            cutoff_radius: 原子间相互作用的截断半径 (Å)
            num_gaussians: 高斯基函数的数量
            use_knn: 是否使用KNN连边方式
            k: KNN中的k值（每个原子连接最近的k个原子）
            interface_cutoff: 定义复合物界面的距离阈值 (Å) - 蛋白质原子到配体的最近距离
        """
        self.cutoff_radius = cutoff_radius
        self.num_gaussians = num_gaussians
        self.use_knn = use_knn
        self.k = k
        self.interface_cutoff = interface_cutoff
        
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
        mol = None
        
        # 尝试多种方法读取SDF文件
        try:
            # 方法1：标准方式
            mol = Chem.MolFromMolFile(sdf_file)
        except Exception as e:
            print(f"[警告] 标准方式读取SDF失败: {e}")
        
        if mol is None:
            try:
                # 方法2：禁用清理，容忍格式问题
                mol = Chem.MolFromMolFile(sdf_file, sanitize=False)
                if mol is not None:
                    # 手动清理分子
                    Chem.SanitizeMol(mol, Chem.SANITIZE_ALL^Chem.SANITIZE_PROPERTIES)
                    print("[信息] 使用容错模式成功读取SDF")
            except Exception as e:
                print(f"[警告] 容错模式读取SDF失败: {e}")
        
        if mol is None:
            try:
                # 方法3：尝试从MOL2文件读取（如果存在）
                mol2_file = sdf_file.replace('.sdf', '.mol2')
                if os.path.exists(mol2_file):
                    mol = Chem.MolFromMol2File(mol2_file)
                    print(f"[信息] 从MOL2文件读取成功: {mol2_file}")
            except Exception as e:
                print(f"[警告] MOL2文件读取失败: {e}")
        
        if mol is None:
            print(f"[错误] 所有方法都无法读取配体文件: {sdf_file}")
            return pd.DataFrame()
        
        # 添加氢原子（如果需要）并重新计算
        try:
            mol = Chem.AddHs(mol)
        except Exception as e:
            print(f"[警告] 添加氢原子失败，使用原分子: {e}")
            # 如果添加氢原子失败，继续使用原分子
        
        # 确保分子有坐标
        if mol.GetNumConformers() == 0:
            print(f"[错误] 分子没有3D坐标信息: {sdf_file}")
            return pd.DataFrame()
            
        try:
            conf = mol.GetConformer()
        except Exception as e:
            print(f"[错误] 无法获取分子坐标: {e}")
            return pd.DataFrame()
            
        atoms_data = []
        
        # 手动统计每个重原子的氢原子邻居数
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != 'H':  # 只保留重原子
                try:
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    
                    # 手动统计氢原子邻居数量
                    hydrogen_count = 0
                    heavy_neighbors = 0
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'H':
                            hydrogen_count += 1
                        else:
                            heavy_neighbors += 1
                    
                    atoms_data.append({
                        'atom_idx': atom.GetIdx(),
                        'element': atom.GetSymbol(),
                        'x': float(pos.x),
                        'y': float(pos.y),
                        'z': float(pos.z),
                        'hydrogen_count': hydrogen_count,
                        'heavy_neighbors': heavy_neighbors,
                        'is_ligand': True
                    })
                except Exception as e:
                    print(f"[警告] 跳过原子 {atom.GetIdx()}: {e}")
                    continue
        
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
    
    def build_knn_edges(self, positions: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用KNN方法构建边连接
        
        Args:
            positions: [num_atoms, 3] 原子坐标
            k: 每个原子连接最近的k个原子
            
        Returns:
            edge_indices: [num_edges, 2] 边索引
            edge_distances: [num_edges] 对应边的距离
        """
        num_atoms = positions.shape[0]
        
        # 计算距离矩阵
        pos_expanded_i = positions.unsqueeze(1)  # [num_atoms, 1, 3]
        pos_expanded_j = positions.unsqueeze(0)  # [1, num_atoms, 3]
        distance_matrix = torch.norm(pos_expanded_i - pos_expanded_j, dim=2)  # [num_atoms, num_atoms]
        
        # 对每一行（每个原子），找到最近的k个原子（排除自身）
        # 将对角线设为无穷大，避免选择自身
        distance_matrix.fill_diagonal_(float('inf'))
        
        # 找到每个原子的k个最近邻
        _, knn_indices = torch.topk(distance_matrix, k, dim=1, largest=False)  # [num_atoms, k]
        
        # 构建边索引
        source_indices = torch.arange(num_atoms).unsqueeze(1).expand(-1, k)  # [num_atoms, k]
        edge_indices = torch.stack([source_indices.flatten(), knn_indices.flatten()], dim=0).t()  # [num_edges, 2]
        
        # 获取对应的距离
        edge_distances = distance_matrix[source_indices.flatten(), knn_indices.flatten()]  # [num_edges]
        
        return edge_indices, edge_distances
    
    def build_radius_edges(self, positions: torch.Tensor, cutoff_radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用截断半径方法构建边连接
        
        Args:
            positions: [num_atoms, 3] 原子坐标
            cutoff_radius: 截断半径
            
        Returns:
            edge_indices: [num_edges, 2] 边索引
            edge_distances: [num_edges] 对应边的距离
        """
        # 计算距离矩阵
        pos_expanded_i = positions.unsqueeze(1)  # [num_atoms, 1, 3]
        pos_expanded_j = positions.unsqueeze(0)  # [1, num_atoms, 3]
        distance_matrix = torch.norm(pos_expanded_i - pos_expanded_j, dim=2)  # [num_atoms, num_atoms]
        
        # 找到在截断半径内的原子对
        mask = (distance_matrix <= cutoff_radius) & (distance_matrix > 0)  # 排除自身
        edge_indices = torch.nonzero(mask, as_tuple=False)  # [num_edges, 2]
        
        # 获取对应的距离值
        edge_distances = distance_matrix[mask]  # [num_edges]
        
        return edge_indices, edge_distances
    
    def identify_interface_atoms(self, protein_positions: torch.Tensor, ligand_positions: torch.Tensor) -> torch.Tensor:
        """
        识别复合物界面的原子
        
        Args:
            protein_positions: [num_protein_atoms, 3] 蛋白质原子坐标
            ligand_positions: [num_ligand_atoms, 3] 配体原子坐标
            
        Returns:
            interface_mask: [num_protein_atoms] 布尔掩码，True表示界面原子
        """
        num_protein_atoms = protein_positions.shape[0]
        num_ligand_atoms = ligand_positions.shape[0]
        
        # 计算蛋白质原子到所有配体原子的距离
        protein_expanded = protein_positions.unsqueeze(1)  # [num_protein_atoms, 1, 3]
        ligand_expanded = ligand_positions.unsqueeze(0)    # [1, num_ligand_atoms, 3]
        
        # 距离矩阵: [num_protein_atoms, num_ligand_atoms]
        distances = torch.norm(protein_expanded - ligand_expanded, dim=2)
        
        # 找到每个蛋白质原子到配体的最小距离
        min_distances = torch.min(distances, dim=1)[0]  # [num_protein_atoms]
        
        # 界面原子：距离配体在interface_cutoff范围内的蛋白质原子
        interface_mask = min_distances <= self.interface_cutoff
        
        return interface_mask
    
    def filter_interface_atoms(self, all_atoms_df, protein_df, ligand_df) -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
        """
        过滤出界面原子，重新构建原子列表
        
        Returns:
            interface_atoms_df: 界面原子的DataFrame
            interface_positions: [num_interface_atoms, 3] 界面原子坐标
            atom_type_mask: [num_interface_atoms] 原子类型掩码 (0=蛋白质界面原子, 1=配体原子)
        """
        # 提取坐标
        protein_positions = torch.tensor(protein_df[['x', 'y', 'z']].values, dtype=torch.float32)
        ligand_positions = torch.tensor(ligand_df[['x', 'y', 'z']].values, dtype=torch.float32)
        
        # 识别界面蛋白质原子
        protein_interface_mask = self.identify_interface_atoms(protein_positions, ligand_positions)
        
        # 过滤界面蛋白质原子
        interface_protein_df = protein_df[protein_interface_mask.numpy()]
        
        # 合并界面蛋白质原子和所有配体原子
        interface_atoms_df = pd.concat([interface_protein_df, ligand_df], ignore_index=True)
        interface_atoms_df['atom_id'] = range(len(interface_atoms_df))
        
        # 提取界面原子坐标
        interface_positions = torch.tensor(
            interface_atoms_df[['x', 'y', 'z']].values, 
            dtype=torch.float32
        )
        
        # 创建原子类型掩码：0表示蛋白质界面原子，1表示配体原子
        num_interface_protein = len(interface_protein_df)
        num_ligand = len(ligand_df)
        atom_type_mask = torch.cat([
            torch.zeros(num_interface_protein, dtype=torch.long),  # 蛋白质界面原子
            torch.ones(num_ligand, dtype=torch.long)               # 配体原子
        ])
        
        print(f"界面识别结果:")
        print(f"  蛋白质总原子数: {len(protein_df)}")
        print(f"  界面蛋白质原子数: {num_interface_protein}")
        print(f"  配体原子数: {num_ligand}")
        print(f"  界面总原子数: {len(interface_atoms_df)}")
        
        return interface_atoms_df, interface_positions, atom_type_mask
    
    def build_graph(self, complex_id: str, pdb_file: str, sdf_file: str) -> Data:
        """
        构建界面分子图（只包含重原子，专注于复合物界面）
        
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
        
        # 过滤界面原子
        interface_atoms_df, interface_positions, atom_type_mask = self.filter_interface_atoms(
            None, protein_df, ligand_df
        )
        
        # 为界面原子提取特征
        # 重新分离界面蛋白质原子和配体原子用于特征提取
        num_interface_protein = torch.sum(atom_type_mask == 0).item()
        interface_protein_df = interface_atoms_df.iloc[:num_interface_protein]
        interface_ligand_df = interface_atoms_df.iloc[num_interface_protein:]
        
        atom_features = self.get_atom_features(interface_protein_df, interface_ligand_df)
        
        # 构建边连接：只对界面原子进行KNN或截断半径连边
        if self.use_knn:
            edge_indices, edge_distances = self.build_knn_edges(interface_positions, self.k)
            print(f"界面KNN连边 (k={self.k}): 边数={edge_indices.shape[0]}")
        else:
            edge_indices, edge_distances = self.build_radius_edges(interface_positions, self.cutoff_radius)
            print(f"界面截断半径连边 (r={self.cutoff_radius}Å): 边数={edge_indices.shape[0]}")
        
        if edge_indices.shape[0] == 0:
            raise ValueError(f"复合物 {complex_id} 界面没有找到任何相互作用边")
        
        # 使用高斯基函数扩展距离特征
        edge_features = self.gaussian_basis_functions(edge_distances)  # [num_edges, num_gaussians]
        
        # 转换为PyG格式
        edge_index = edge_indices.t()  # [2, num_edges]
        
        # 构建PyTorch Geometric Data对象
        data = Data(
            x=atom_features,           # [num_interface_atoms, atom_feature_dim] 
            edge_index=edge_index,     # [2, num_edges]
            edge_attr=edge_features,   # [num_edges, num_gaussians]
            pos=interface_positions,   # [num_interface_atoms, 3]
            complex_id=complex_id,
            num_atoms=len(interface_atoms_df),
            num_protein_atoms=num_interface_protein,
            num_ligand_atoms=len(interface_ligand_df)
        )
        
        return data
    
    def get_positional_encoding(self, positions: torch.Tensor, d_model: int = 64) -> torch.Tensor:
        """
        为原子坐标生成位置编码
        
        Args:
            positions: 原子坐标 [num_atoms, 3]
            d_model: 编码维度
            
        Returns:
            位置编码 [num_atoms, d_model]
        """
        # 简化版位置编码：使用傅里叶特征
        # 对每个空间维度使用不同的频率
        num_atoms = positions.shape[0]
        pe = torch.zeros(num_atoms, d_model)
        
        # 为每个坐标轴生成不同频率的正弦/余弦编码
        for dim in range(3):  # x, y, z
            coord = positions[:, dim]  # [num_atoms]
            
            # 每个坐标轴使用 d_model//3 个维度
            dim_size = d_model // 3
            if dim == 2:  # 确保总维度是d_model
                dim_size = d_model - 2 * (d_model // 3)
                
            for i in range(dim_size // 2):
                # 使用不同的频率
                freq = 1.0 / (10000 ** (2 * i / dim_size))
                
                pe[:, dim * (d_model // 3) + 2*i] = torch.sin(coord * freq)
                pe[:, dim * (d_model // 3) + 2*i + 1] = torch.cos(coord * freq)
        
        return pe.float()
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """返回特征维度信息"""
        return {
            'node_dim': self.atom_feature_dim,  # 12维: 原子类型(10) + 氢原子数(1) + 重原子邻居数(1)
            'edge_dim': self.num_gaussians,     # 16维: 高斯基函数扩展
            'pos_dim': 3
        }


class Stage2GraphBuilder(MolecularGraphBuilder):
    """阶段二：完整的物理信息增强3D-ESA图构建器"""
    
    def __init__(self, cutoff_radius: float = 5.0, num_gaussians: int = 16, use_knn: bool = False, k: int = 16, interface_cutoff: float = 8.0):
        super().__init__(cutoff_radius, num_gaussians, use_knn, k, interface_cutoff)
        
        # 阶段二特征维度定义
        self.hybridization_types = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED']
        
        # 节点特征维度:
        # 原子类型(10) + 形式电荷(1) + 杂化类型(6) + 芳香性(1) + 重邻居数(1) + 总氢数(1) = 20
        self.atom_feature_dim = 10 + 1 + len(self.hybridization_types) + 1 + 1 + 1
        
        # 边特征维度:
        # 距离GBF(16) + 方向向量(3) + 电荷乘积(1) = 20
        self.edge_feature_dim = self.num_gaussians + 3 + 1

    def get_atom_features(self, all_atoms_df: pd.DataFrame, ligand_mol: Chem.Mol) -> torch.Tensor:
        """
        阶段二：提取完整的原子物理化学特征
        """
        # 为配体原子计算Gasteiger电荷
        AllChem.ComputeGasteigerCharges(ligand_mol)
        
        all_features = []
        
        # 遍历DataFrame中的所有原子（已合并蛋白质和配体）
        for _, row in all_atoms_df.iterrows():
            is_ligand = row.get('is_ligand', False)
            
            if is_ligand is True:
                atom = ligand_mol.GetAtomWithIdx(int(row['atom_idx']))
                
                # 1. 原子类型 (10维)
                atom_type_oh = self.get_atom_type_onehot(atom.GetSymbol())
                
                # 2. 形式电荷 (1维)
                formal_charge = float(atom.GetFormalCharge())
                
                # 3. 杂化类型 (6维)
                hybridization = str(atom.GetHybridization())
                hybrid_oh = np.zeros(len(self.hybridization_types))
                if hybridization in self.hybridization_types:
                    hybrid_oh[self.hybridization_types.index(hybridization)] = 1.0
                else:
                    hybrid_oh[-1] = 1.0 # UNSPECIFIED
                
                # 4. 芳香性 (1维)
                is_aromatic = float(atom.GetIsAromatic())
                
                # 5. 重原子邻居数 (1维)
                heavy_neighbors = float(sum(1 for n in atom.GetNeighbors() if n.GetSymbol() != 'H'))
                
                # 6. 总氢原子数 (1维)
                total_hydrogens = float(atom.GetTotalNumHs())
                
                # Gasteiger电荷（用于边特征计算，这里先存起来）
                gasteiger_charge = float(atom.GetProp('_GasteigerCharge'))
                all_atoms_df.loc[row.name, 'gasteiger_charge'] = gasteiger_charge

            else: # 蛋白质原子
                # 对蛋白质原子使用简化特征
                atom_type_oh = self.get_atom_type_onehot(row['element'])
                formal_charge = 0.0
                hybrid_oh = np.zeros(len(self.hybridization_types))
                hybrid_oh[-1] = 1.0 # UNSPECIFIED
                is_aromatic = 0.0
                heavy_neighbors = self._estimate_heavy_neighbors(row['element'])
                total_hydrogens = 0.0 # PDB中不含H
                all_atoms_df.loc[row.name, 'gasteiger_charge'] = 0.0 # 简化处理

            features = np.concatenate([
                atom_type_oh,
                [formal_charge],
                hybrid_oh,
                [is_aromatic],
                [heavy_neighbors],
                [total_hydrogens]
            ])
            all_features.append(features)
            
        return torch.tensor(np.array(all_features), dtype=torch.float32)

    def get_full_edge_features(self, edge_indices: torch.Tensor, positions: torch.Tensor, all_atoms_df: pd.DataFrame) -> torch.Tensor:
        """
        阶段二：计算完整的边特征（距离、方向、电荷）
        """
        # 1. 距离特征 (高斯基函数)
        src, dst = edge_indices[0], edge_indices[1]
        distances = torch.norm(positions[dst] - positions[src], dim=-1)
        distance_features = self.gaussian_basis_functions(distances)
        
        # 2. 方向特征 (归一化向量)
        direction_vectors = (positions[dst] - positions[src]) / (distances.unsqueeze(-1) + 1e-8)
        
        # 3. 电荷相互作用特征
        gasteiger_charges = torch.tensor(all_atoms_df['gasteiger_charge'].values, dtype=torch.float32).to(edge_indices.device)
        charge_product = (gasteiger_charges[src] * gasteiger_charges[dst]).unsqueeze(-1)
        
        # 拼接所有边特征
        full_edge_features = torch.cat([distance_features, direction_vectors, charge_product], dim=1)
        
        return full_edge_features

    def build_graph(self, complex_id: str, pdb_file: str, sdf_file: str) -> Data:
        """
        阶段二：构建包含完整物理信息的界面分子图
        """
        # 解析文件
        protein_df = self.parse_pdb_file(pdb_file)
        ligand_df = self.parse_sdf_ligand(sdf_file) # 这会返回带RDKit信息的df
        
        # 加载RDKit Mol对象
        ligand_mol = Chem.MolFromMolFile(sdf_file, sanitize=True)
        if ligand_mol is None: # 容错
             ligand_mol = Chem.MolFromMolFile(sdf_file, sanitize=False)
             Chem.SanitizeMol(ligand_mol, Chem.SANITIZE_ALL^Chem.SANITIZE_PROPERTIES)
        ligand_mol = Chem.AddHs(ligand_mol, addCoords=True)

        if protein_df.empty or ligand_df.empty or ligand_mol is None:
            raise ValueError(f"无法解析PDB/SDF文件: {complex_id}")
        
        # 过滤界面原子
        interface_atoms_df, interface_positions, _ = self.filter_interface_atoms(
            None, protein_df, ligand_df
        )
        
        # 提取完整的原子特征
        atom_features = self.get_atom_features(interface_atoms_df, ligand_mol)
        
        # 构建边
        if self.use_knn:
            edge_indices_t, _ = self.build_knn_edges(interface_positions, self.k)
        else:
            edge_indices_t, _ = self.build_radius_edges(interface_positions, self.cutoff_radius)
        
        edge_index = edge_indices_t.t()
        if edge_index.shape[1] == 0:
            raise ValueError(f"复合物 {complex_id} 界面没有找到任何相互作用边")
            
        # 提取完整的边特征
        edge_features = self.get_full_edge_features(edge_index, interface_positions, interface_atoms_df)
        
        data = Data(
            x=atom_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            pos=interface_positions,
            complex_id=complex_id,
            num_atoms=interface_positions.shape[0]
        )
        
        return data

    def get_feature_dimensions(self) -> Dict[str, int]:
        """返回阶段二的特征维度信息"""
        return {
            'node_dim': self.atom_feature_dim,
            'edge_dim': self.edge_feature_dim,
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
            
            # 检查边特征的有效性
            print(f"\n边特征检查:")
            print(f"  边特征形状: {data.edge_attr.shape}")
            print(f"  边特征范围: min={data.edge_attr.min().item():.4f}, max={data.edge_attr.max().item():.4f}")
            
            # 检查前几条边的距离和特征
            print(f"\n前5条边的信息:")
            for i in range(min(5, data.edge_index.shape[1])):
                src_idx = data.edge_index[0, i].item()
                tgt_idx = data.edge_index[1, i].item()
                
                # 计算真实距离
                src_pos = data.pos[src_idx]
                tgt_pos = data.pos[tgt_idx]
                true_distance = torch.norm(tgt_pos - src_pos).item()
                
                # 获取边特征（高斯基函数的响应）
                edge_features = data.edge_attr[i]
                
                print(f"  边{i}: 原子{src_idx}->{tgt_idx}, "
                      f"距离={true_distance:.3f}Å, "
                      f"高斯特征前3维={edge_features[:3].tolist()}")
            
            # 检查一些配体原子的特征
            ligand_start_idx = data.num_protein_atoms
            if data.num_ligand_atoms > 0:
                print("\n配体原子特征示例:")
                for i in range(min(5, data.num_ligand_atoms)):
                    idx = ligand_start_idx + i
                    features = data.x[idx]
                    # 解析特征
                    atom_type = int(torch.argmax(features[:10]).item())
                    hydrogen_count = features[10].item()
                    heavy_neighbors = features[11].item()
                    
                    atom_names = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Other']
                    print(f"  原子{i}: 类型={atom_names[atom_type]}, "
                          f"氢原子数={hydrogen_count:.0f}, "
                          f"重原子邻居数={heavy_neighbors:.0f}")
            
            # 验证高斯基函数是否工作正常
            print(f"\n高斯基函数测试:")
            test_distances = torch.tensor([1.0, 2.5, 4.0, 5.0])
            for dist in test_distances:
                gaussian_features = builder.gaussian_basis_functions(dist.unsqueeze(0))[0]
                max_response_idx = int(torch.argmax(gaussian_features).item())
                max_response = gaussian_features[max_response_idx].item()
                corresponding_center = builder.gaussian_centers[max_response_idx].item()
                print(f"  距离{dist:.1f}Å -> 最强响应在中心{corresponding_center:.2f}Å (响应值={max_response:.3f})")
            
        except Exception as e:
            print(f"构建图时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"找不到文件: {pdb_file} 或 {sdf_file}")

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
