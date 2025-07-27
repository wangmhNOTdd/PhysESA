import numpy as np
import pandas as pd
import torch
from Bio import PDB
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch.nn import functional as F
import json
import os
from typing import Tuple, Dict, List, Optional

from rdkit.Chem import BRICS
from collections import defaultdict

class MultiScaleGraphBuilder:
    """
    多尺度、物理信息增强的3D图构建器。
    该类负责：
    1. 定义新的复合物界面（基于残基距离）。
    2. 构建一个细粒度的原子图（atom-level graph）。
    3. 构建一个粗粒度的官能团/残基图（coarse-grained graph）。
    4. 提供从原子到粗粒度节点的映射，用于多尺度模型。
    """
    
    def __init__(self, cutoff_radius: float = 5.0, num_gaussians: int = 16, use_knn: bool = False, k: int = 16, interface_cutoff: float = 10.0):
        """
        Args:
            cutoff_radius: 原子间相互作用的截断半径 (Å)，用于半径连边模式。
            num_gaussians: 用于距离编码的高斯基函数的数量。
            use_knn: 是否使用KNN方法构建原子图的边。
            k: 在KNN模式下，每个原子连接的最近邻居数量。
            interface_cutoff: 定义蛋白质-配体界面的距离阈值 (Å)，作用于残基级别。
        """
        self.cutoff_radius = cutoff_radius
        self.num_gaussians = num_gaussians
        self.use_knn = use_knn
        self.k = k
        self.interface_cutoff = interface_cutoff
        
        self.gaussian_centers = torch.linspace(0, cutoff_radius, num_gaussians)
        self.gaussian_width = (cutoff_radius / num_gaussians) * 0.5

        self.hybridization_types = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED']
        self.atom_feature_dim = 10 + 1 + len(self.hybridization_types) + 1 + 1 + 1
        self.edge_feature_dim = self.num_gaussians + 3 + 1

        self.pdb_parser = PDB.PDBParser(QUIET=True)


    def _get_ligand_motifs(self, ligand_mol: Chem.Mol) -> Tuple[List[List[int]], List[str]]:
        """使用BRICS分解配体，返回每个motif包含的原子索引和motif类型。"""
        brics_bonds = list(BRICS.FindBRICSBonds(ligand_mol))
        if not brics_bonds: # 如果分子无法被分解，则整个分子作为一个motif
            return [[atom.GetIdx() for atom in ligand_mol.GetAtoms()]], ["LIG"]

        # 修复：正确获取要断开的键的索引
        bonds_to_break = []
        for bond_info in brics_bonds:
            atom_indices = bond_info[0]
            bond = ligand_mol.GetBondBetweenAtoms(atom_indices[0], atom_indices[1])
            if bond is not None:
                bonds_to_break.append(bond.GetIdx())
        
        if not bonds_to_break: # 如果没有找到有效的键来断开
             return [[atom.GetIdx() for atom in ligand_mol.GetAtoms()]], ["LIG"]

        broken_mol = Chem.FragmentOnBonds(ligand_mol, bonds_to_break, addDummies=False)
        motif_atom_indices = Chem.GetMolFrags(broken_mol, asMols=False)
        
        motif_ids = [f"LIG_MOTIF_{i}" for i in range(len(motif_atom_indices))]
        return list(motif_atom_indices), motif_ids

    def parse_protein_with_biopython(self, pdb_file: str) -> Optional[pd.DataFrame]:
        """
        使用BioPython解析PDB文件，以获得最完整、最原始的原子和残基信息。
        这是确保不错过任何界面原子的关键。
        """
        try:
            structure = self.pdb_parser.get_structure("protein", pdb_file)
            atoms_data = []
            for atom in structure.get_atoms():
                # 跳过氢原子和水分子中的原子
                if atom.element == 'H' or atom.get_parent().get_resname() == 'HOH':
                    continue
                
                res_id = f"{atom.get_parent().get_parent().id}_{atom.get_parent().get_resname()}_{atom.get_parent().get_id()[1]}"
                
                atoms_data.append({
                    'atom_serial_number': atom.get_serial_number(),
                    'element': atom.element,
                    'x': atom.coord[0], 'y': atom.coord[1], 'z': atom.coord[2],
                    'res_id': res_id,
                    'is_ligand': False
                })
            return pd.DataFrame(atoms_data)
        except Exception as e:
            print(f"[错误] BioPython解析PDB失败: {os.path.basename(pdb_file)}, {e}")
            return None

    def parse_sdf_ligand(self, sdf_file: str) -> Tuple[Optional[pd.DataFrame], Optional[Chem.Mol]]:
        """解析SDF文件，提取配体重原子信息。"""
        mol = Chem.MolFromMolFile(sdf_file, sanitize=True, removeHs=False)
        if mol is None: return None, None

        try:
            mol = Chem.AddHs(mol, addCoords=True)
        except Exception:
            pass

        if mol.GetNumConformers() == 0: return None, None
            
        conf = mol.GetConformer()
        atoms_data = []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            atoms_data.append({
                'atom_idx_rdkit': atom.GetIdx(),
                'element': atom.GetSymbol(),
                'x': float(pos.x), 'y': float(pos.y), 'z': float(pos.z),
                'is_ligand': True
            })
        
        return pd.DataFrame(atoms_data), mol

    def get_atom_type_onehot(self, symbol: str) -> np.ndarray:
        """原子类型独热编码。"""
        elements = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Other']
        onehot = np.zeros(len(elements))
        try:
            idx = elements.index(symbol)
        except ValueError:
            idx = elements.index('Other')
        onehot[idx] = 1.0
        return onehot

    def get_simplified_protein_features(self, protein_df: pd.DataFrame) -> pd.DataFrame:
        """为蛋白质原子生成简化的、不依赖RDKit的特征。"""
        # 特征作为新的列添加到DataFrame中，以便后续处理
        
        # 1. 原子类型 (10维)
        elements = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Other']
        atom_type_map = {elem: i for i, elem in enumerate(elements)}
        protein_df['atom_type_idx'] = protein_df['element'].map(lambda x: atom_type_map.get(x, atom_type_map['Other']))

        # 2. 残基类型 (21维)
        residue_types = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                         'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'TRP', 'TYR', 'THR', 'VAL', 'UNK']
        res_map = {res: i for i, res in enumerate(residue_types)}
        protein_df['res_type_idx'] = protein_df['res_id'].map(lambda x: res_map.get(x.split('_')[1], res_map['UNK']))
        
        # Gasteiger电荷的粗略估计（对于边特征）
        protein_df['gasteiger_charge'] = 0.0 # 简化处理，设为0

        return protein_df

    def get_ligand_features(self, ligand_df: pd.DataFrame, ligand_mol: Chem.Mol) -> pd.DataFrame:
        """为配体原子生成完整的RDKit化学特征，并附加到DataFrame。"""
        AllChem.ComputeGasteigerCharges(ligand_mol)
        
        features_list = []
        for _, row in ligand_df.iterrows():
            atom = ligand_mol.GetAtomWithIdx(int(row['atom_idx_rdkit']))
            
            features = {
                'formal_charge': float(atom.GetFormalCharge()),
                'is_aromatic': float(atom.GetIsAromatic()),
                'heavy_neighbors': float(sum(1 for n in atom.GetNeighbors() if n.GetSymbol() != 'H')),
                'total_hydrogens': float(atom.GetTotalNumHs())
            }
            
            hybridization = str(atom.GetHybridization())
            for h_type in self.hybridization_types:
                features[f'hybrid_{h_type}'] = 1.0 if hybridization == h_type else 0.0
            
            gasteiger_charge = float(atom.GetProp('_GasteigerCharge'))
            features['gasteiger_charge'] = gasteiger_charge if np.isfinite(gasteiger_charge) else 0.0
            
            features_list.append(features)
            
        ligand_features_df = pd.DataFrame(features_list, index=ligand_df.index)
        return pd.concat([ligand_df, ligand_features_df], axis=1)

    def gaussian_basis_functions(self, distances: torch.Tensor) -> torch.Tensor:
        """使用高斯基函数扩展距离特征。"""
        distances = distances.unsqueeze(-1)
        centers = self.gaussian_centers.to(distances.device).unsqueeze(0)
        return torch.exp(-((distances - centers) ** 2) / (self.gaussian_width ** 2))

    def build_edges(self, positions: torch.Tensor) -> torch.Tensor:
        """根据配置（KNN或半径）构建边。"""
        if self.use_knn:
            dist_matrix = torch.cdist(positions, positions)
            dist_matrix.fill_diagonal_(float('inf'))
            if positions.shape[0] <= self.k:
                k = positions.shape[0] - 1
            else:
                k = self.k
            if k <= 0:
                return torch.empty((2, 0), dtype=torch.long)
            _, knn_indices = torch.topk(dist_matrix, k, dim=1, largest=False)
            source_indices = torch.arange(positions.shape[0]).unsqueeze(1).expand(-1, knn_indices.shape[1])
            return torch.stack([source_indices.flatten(), knn_indices.flatten()], dim=0)
        else:
            dist_matrix = torch.cdist(positions, positions)
            mask = (dist_matrix <= self.cutoff_radius) & (dist_matrix > 0)
            return torch.nonzero(mask).t()

    def get_full_edge_features(self, edge_index: torch.Tensor, positions: torch.Tensor, all_atoms_df: pd.DataFrame) -> torch.Tensor:
        """计算完整的边特征（距离、方向、电荷）。"""
        src, dst = edge_index[0], edge_index[1]
        dist = torch.norm(positions[dst] - positions[src], dim=-1)
        
        dist_features = self.gaussian_basis_functions(dist)
        direction_vectors = (positions[dst] - positions[src]) / (dist.unsqueeze(-1) + 1e-8)
        
        charges = torch.tensor(all_atoms_df['gasteiger_charge'].values, dtype=torch.float32).to(edge_index.device)
        charge_product = (charges[src] * charges[dst]).unsqueeze(-1)
        
        return torch.cat([dist_features, direction_vectors, charge_product], dim=1)

    def build_graph(self, complex_id: str, pdb_file: str, sdf_file: str) -> Optional[Data]:
        """构建包含多尺度信息的图对象。"""
        # 1. 解析分子
        protein_df_full = self.parse_protein_with_biopython(pdb_file)
        ligand_df_raw, ligand_mol = self.parse_sdf_ligand(sdf_file)
        if protein_df_full is None or ligand_df_raw is None or ligand_mol is None: return None

        # 2. 定义界面
        protein_pos = torch.tensor(protein_df_full[['x', 'y', 'z']].values, dtype=torch.float32)
        ligand_pos = torch.tensor(ligand_df_raw[['x', 'y', 'z']].values, dtype=torch.float32)

        dist_matrix = torch.cdist(protein_pos, ligand_pos)
        min_dist_to_ligand, _ = torch.min(dist_matrix, dim=1)
        
        interface_atom_mask = (min_dist_to_ligand <= self.interface_cutoff).numpy()
        interface_residues = protein_df_full.loc[interface_atom_mask, 'res_id']
        if interface_residues.empty: return None
        interface_res_ids = interface_residues.unique()
        
        protein_df = protein_df_full[protein_df_full['res_id'].isin(interface_res_ids)].copy()
        if protein_df.empty: return None

        # 3. 准备细粒度图（原子图）
        protein_df = self.get_simplified_protein_features(protein_df)
        ligand_df = self.get_ligand_features(ligand_df_raw, ligand_mol)
        
        # 合并DataFrame并创建最终的特征张量
        atom_df = pd.concat([protein_df, ligand_df], ignore_index=True).reset_index(drop=True)
        atom_df.fillna(0, inplace=True) # 用0填充所有NaN

        # One-hot编码
        atom_type_oh = F.one_hot(torch.tensor(atom_df['atom_type_idx'].values, dtype=torch.long), num_classes=10)
        res_type_oh = F.one_hot(torch.tensor(atom_df['res_type_idx'].values, dtype=torch.long), num_classes=21)
        
        # 数值特征
        numeric_features = torch.tensor(atom_df[[
            'formal_charge', 'is_aromatic', 'heavy_neighbors', 'total_hydrogens',
            'hybrid_SP', 'hybrid_SP2', 'hybrid_SP3', 'hybrid_SP3D', 'hybrid_SP3D2', 'hybrid_UNSPECIFIED'
        ]].values, dtype=torch.float32)

        atom_features = torch.cat([atom_type_oh, res_type_oh, numeric_features], dim=1)
        atom_pos = torch.tensor(atom_df[['x', 'y', 'z']].values, dtype=torch.float32)
        atom_edge_index = self.build_edges(atom_pos)
        if atom_edge_index.shape[1] == 0: return None
        atom_edge_features = self.get_full_edge_features(atom_edge_index, atom_pos, atom_df)

        # 4. 准备粗粒度图（残基/官能团图）
        ligand_motifs, ligand_motif_ids = self._get_ligand_motifs(ligand_mol)
        
        # 创建从原子到粗粒度节点的映射
        coarse_node_map = {} # "res_id" or "motif_id" -> new_coarse_idx
        atom_to_coarse_idx = []
        
        # 处理蛋白质残基
        for res_id in interface_res_ids:
            if res_id not in coarse_node_map:
                coarse_node_map[res_id] = len(coarse_node_map)
        
        # 处理配体官能团
        ligand_rdkit_idx_to_motif_id = {}
        for i, motif_atoms in enumerate(ligand_motifs):
            motif_id = ligand_motif_ids[i]
            if motif_id not in coarse_node_map:
                coarse_node_map[motif_id] = len(coarse_node_map)
            for atom_idx in motif_atoms:
                ligand_rdkit_idx_to_motif_id[atom_idx] = motif_id

        # 填充映射向量
        for _, row in atom_df.iterrows():
            if row['is_ligand']:
                motif_id = ligand_rdkit_idx_to_motif_id[row['atom_idx_rdkit']]
                atom_to_coarse_idx.append(coarse_node_map[motif_id])
            else:
                atom_to_coarse_idx.append(coarse_node_map[row['res_id']])
        
        atom_to_coarse_idx = torch.tensor(atom_to_coarse_idx, dtype=torch.long)

        # 构建粗粒度图的节点和边
        num_coarse_nodes = len(coarse_node_map)
        coarse_pos = torch.zeros(num_coarse_nodes, 3, dtype=torch.float32)
        coarse_pos.index_add_(0, atom_to_coarse_idx, atom_pos)
        
        counts = torch.zeros(num_coarse_nodes, 1, dtype=torch.float32)
        counts.index_add_(0, atom_to_coarse_idx, torch.ones_like(atom_pos[:, :1]))
        coarse_pos = coarse_pos / counts.clamp(min=1)

        coarse_edge_index = self.build_edges(coarse_pos) # 在粗粒度节点质心上建边

        # 5. 整合到Data对象
        data = Data(
            x=atom_features,
            edge_index=atom_edge_index,
            edge_attr=atom_edge_features,
            pos=atom_pos,
            complex_id=complex_id,
            num_nodes=atom_features.shape[0], # 兼容旧版
            
            # 多尺度信息
            atom_to_coarse_idx=atom_to_coarse_idx,
            coarse_pos=coarse_pos,
            coarse_edge_index=coarse_edge_index,
            num_coarse_nodes=num_coarse_nodes
        )

        # 最终检查
        for key, value in data.items():
            if torch.is_tensor(value) and (torch.isnan(value).any() or torch.isinf(value).any()):
                print(f"[错误] 跳过 {complex_id}: 数据字段 '{key}' 包含无效值。")
                return None
        
        return data

    def get_feature_dimensions(self) -> Dict[str, int]:
        """返回特征维度信息。"""
        # Atom type (10) + Res type (21) + Numeric (10)
        node_dim = 10 + 21 + 10
        return {
            'node_dim': node_dim,
            'edge_dim': self.edge_feature_dim,
            'pos_dim': 3
        }


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
