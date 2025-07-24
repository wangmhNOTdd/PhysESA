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

class GraphBuilder:
    """
    统一的、物理信息增强的3D-ESA图构建器。
    该类整合了从PDB和SDF文件解析分子、识别复合物界面、
    提取原子和边特征以及构建PyTorch Geometric图对象的完整流程。
    """
    
    def __init__(self, cutoff_radius: float = 5.0, num_gaussians: int = 16, use_knn: bool = False, k: int = 16, interface_cutoff: float = 8.0):
        """
        Args:
            cutoff_radius: 原子间相互作用的截断半径 (Å)，用于半径连边模式。
            num_gaussians: 用于距离编码的高斯基函数的数量。
            use_knn: 是否使用KNN（K-Nearest Neighbors）方法构建图的边。
            k: 在KNN模式下，每个原子连接的最近邻居数量。
            interface_cutoff: 定义蛋白质-配体界面的距离阈值 (Å)。
        """
        self.cutoff_radius = cutoff_radius
        self.num_gaussians = num_gaussians
        self.use_knn = use_knn
        self.k = k
        self.interface_cutoff = interface_cutoff
        
        # 高斯基函数参数：从0到cutoff_radius均匀分布
        self.gaussian_centers = torch.linspace(0, cutoff_radius, num_gaussians)
        self.gaussian_width = (cutoff_radius / num_gaussians) * 0.5  # β参数

        # 特征维度定义
        self.hybridization_types = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED']
        
        # 节点特征维度:
        # 原子类型(10) + 形式电荷(1) + 杂化类型(6) + 芳香性(1) + 重邻居数(1) + 总氢数(1) = 20
        self.atom_feature_dim = 10 + 1 + len(self.hybridization_types) + 1 + 1 + 1
        
        # 边特征维度:
        # 距离GBF(num_gaussians) + 方向向量(3) + 电荷乘积(1)
        self.edge_feature_dim = self.num_gaussians + 3 + 1

    def parse_pdb_file(self, pdb_file: str) -> pd.DataFrame:
        """解析PDB文件，提取重原子信息。"""
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('complex', pdb_file)
        
        atoms_data = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        element = atom.element.strip() if atom.element else 'C'
                        if element != 'H': # 只保留重原子
                            coord = atom.get_coord()
                            atoms_data.append({
                                'element': element,
                                'x': coord[0], 'y': coord[1], 'z': coord[2],
                                'atom_name': atom.get_name(),
                                'residue_name': residue.get_resname(),
                                'chain_id': chain.id
                            })
        return pd.DataFrame(atoms_data)

    def parse_sdf_ligand(self, sdf_file: str) -> Tuple[Optional[pd.DataFrame], Optional[Chem.Mol]]:
        """
        解析SDF文件，提取配体重原子信息，并返回RDKit Mol对象。
        增加了对常见SDF格式错误的鲁棒性处理。
        """
        # RDKit的错误处理：MolFromMolFile在失败时返回None并打印到stderr，而不是抛出异常。
        # 因此，我们不需要try-except块，而是检查返回值。
        
        # 1. 尝试标准、严格的解析
        mol = Chem.MolFromMolFile(sdf_file, sanitize=True)
        
        # 2. 如果失败，尝试无消毒的解析，然后手动进行安全的消毒
        if mol is None:
            # print(f"[信息] 标准解析失败: {os.path.basename(sdf_file)}。尝试容错模式...")
            mol = Chem.MolFromMolFile(sdf_file, sanitize=False)
            if mol is not None:
                try:
                    # 手动执行一系列“安全”的消毒操作。
                    # SANITIZE_PROPERTIES 是导致价键错误的主要原因，我们将其排除。
                    safe_sanitize_ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
                    Chem.SanitizeMol(mol, safe_sanitize_ops)
                except Exception as e:
                    # 如果手动消毒也失败，则放弃该分子。
                    print(f"[错误] 手动消毒失败: {os.path.basename(sdf_file)}, 错误: {e}")
                    return None, None

        # 3. 如果所有方法都失败，则返回None
        if mol is None:
            return None, None

        # 4. 为成功解析的分子添加氢原子
        try:
            mol = Chem.AddHs(mol, addCoords=True)
        except Exception as e:
            # 有时添加氢原子也会失败
            print(f"[警告] 添加氢原子失败: {os.path.basename(sdf_file)}, {e}")

        # 5. 检查3D坐标
        if mol.GetNumConformers() == 0:
            print(f"[警告] 分子没有3D坐标信息: {os.path.basename(sdf_file)}")
            return None, None
            
        # 6. 提取原子数据
        conf = mol.GetConformer()
        atoms_data = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != 'H':
                pos = conf.GetAtomPosition(atom.GetIdx())
                atoms_data.append({
                    'atom_idx': atom.GetIdx(),
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

    def _estimate_heavy_neighbors(self, element: str) -> float:
        """为PDB文件中的原子估计重原子邻居数。"""
        estimates = {'C': 2.0, 'N': 1.5, 'O': 1.0, 'S': 2.0, 'P': 3.0}
        return estimates.get(element, 1.0)

    def get_atom_features(self, all_atoms_df: pd.DataFrame, ligand_mol: Chem.Mol) -> torch.Tensor:
        """提取完整的原子物理化学特征。"""
        AllChem.ComputeGasteigerCharges(ligand_mol)
        
        all_features = []
        for _, row in all_atoms_df.iterrows():
            if row.get('is_ligand', False):
                atom = ligand_mol.GetAtomWithIdx(int(row['atom_idx']))
                atom_type_oh = self.get_atom_type_onehot(atom.GetSymbol())
                formal_charge = float(atom.GetFormalCharge())
                hybridization = str(atom.GetHybridization())
                hybrid_oh = np.zeros(len(self.hybridization_types))
                if hybridization in self.hybridization_types:
                    hybrid_oh[self.hybridization_types.index(hybridization)] = 1.0
                else:
                    hybrid_oh[-1] = 1.0 # UNSPECIFIED
                is_aromatic = float(atom.GetIsAromatic())
                heavy_neighbors = float(sum(1 for n in atom.GetNeighbors() if n.GetSymbol() != 'H'))
                total_hydrogens = float(atom.GetTotalNumHs())
                gasteiger_charge = float(atom.GetProp('_GasteigerCharge'))
                all_atoms_df.at[row.name, 'gasteiger_charge'] = gasteiger_charge
            else: # 蛋白质原子
                atom_type_oh = self.get_atom_type_onehot(row['element'])
                formal_charge = 0.0
                hybrid_oh = np.zeros(len(self.hybridization_types)); hybrid_oh[-1] = 1.0
                is_aromatic = 0.0
                heavy_neighbors = self._estimate_heavy_neighbors(row['element'])
                total_hydrogens = 0.0
                all_atoms_df.at[row.name, 'gasteiger_charge'] = 0.0

            features = np.concatenate([
                atom_type_oh, [formal_charge], hybrid_oh,
                [is_aromatic], [heavy_neighbors], [total_hydrogens]
            ])
            all_features.append(features)
            
        return torch.tensor(np.array(all_features), dtype=torch.float32)

    def gaussian_basis_functions(self, distances: torch.Tensor) -> torch.Tensor:
        """使用高斯基函数扩展距离特征。"""
        distances = distances.unsqueeze(-1)
        centers = self.gaussian_centers.to(distances.device).unsqueeze(0)
        return torch.exp(-((distances - centers) ** 2) / (self.gaussian_width ** 2))

    def build_edges(self, positions: torch.Tensor) -> torch.Tensor:
        """根据配置（KNN或半径）构建边。"""
        if self.use_knn:
            return self.build_knn_edges(positions, self.k)
        else:
            return self.build_radius_edges(positions, self.cutoff_radius)

    def build_knn_edges(self, positions: torch.Tensor, k: int) -> torch.Tensor:
        """使用KNN方法构建边索引。"""
        num_atoms = positions.shape[0]
        dist_matrix = torch.cdist(positions, positions)
        dist_matrix.fill_diagonal_(float('inf'))
        _, knn_indices = torch.topk(dist_matrix, k, dim=1, largest=False)
        source_indices = torch.arange(num_atoms).unsqueeze(1).expand(-1, k)
        return torch.stack([source_indices.flatten(), knn_indices.flatten()], dim=0)

    def build_radius_edges(self, positions: torch.Tensor, cutoff_radius: float) -> torch.Tensor:
        """使用截断半径方法构建边索引。"""
        dist_matrix = torch.cdist(positions, positions)
        mask = (dist_matrix <= cutoff_radius) & (dist_matrix > 0)
        return torch.nonzero(mask).t()

    def identify_interface_atoms(self, protein_positions: torch.Tensor, ligand_positions: torch.Tensor) -> torch.Tensor:
        """识别复合物界面的蛋白质原子。"""
        if protein_positions.nelement() == 0 or ligand_positions.nelement() == 0:
            return torch.tensor([], dtype=torch.bool)
        dist_matrix = torch.cdist(protein_positions, ligand_positions)
        min_distances, _ = torch.min(dist_matrix, dim=1)
        return min_distances <= self.interface_cutoff

    def filter_interface_atoms(self, protein_df: pd.DataFrame, ligand_df: pd.DataFrame) -> Tuple[pd.DataFrame, torch.Tensor]:
        """过滤出界面原子，返回包含所有界面原子的DataFrame和坐标。"""
        protein_pos = torch.tensor(protein_df[['x', 'y', 'z']].values, dtype=torch.float32)
        ligand_pos = torch.tensor(ligand_df[['x', 'y', 'z']].values, dtype=torch.float32)
        
        protein_interface_mask = self.identify_interface_atoms(protein_pos, ligand_pos)
        interface_protein_df = protein_df[protein_interface_mask.numpy()]
        
        interface_atoms_df = pd.concat([interface_protein_df, ligand_df], ignore_index=True)
        interface_positions = torch.tensor(interface_atoms_df[['x', 'y', 'z']].values, dtype=torch.float32)
        
        return interface_atoms_df, interface_positions

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
        """构建包含完整物理信息的界面分子图。"""
        protein_df = self.parse_pdb_file(pdb_file)
        ligand_df, ligand_mol = self.parse_sdf_ligand(sdf_file)

        if protein_df.empty or ligand_df is None or ligand_df.empty or ligand_mol is None:
            print(f"[警告] 跳过 {complex_id}: 无法解析PDB/SDF文件。")
            return None
        
        interface_atoms_df, interface_pos = self.filter_interface_atoms(protein_df, ligand_df)
        
        if interface_pos.shape[0] == 0:
            print(f"[警告] 跳过 {complex_id}: 界面未发现原子。")
            return None

        atom_features = self.get_atom_features(interface_atoms_df, ligand_mol)
        edge_index = self.build_edges(interface_pos)
        
        if edge_index.shape[1] == 0:
            print(f"[警告] 跳过 {complex_id}: 界面未发现相互作用边。")
            return None
            
        edge_features = self.get_full_edge_features(edge_index, interface_pos, interface_atoms_df)
        
        return Data(
            x=atom_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            pos=interface_pos,
            complex_id=complex_id,
            num_atoms=interface_pos.shape[0]
        )

    def get_feature_dimensions(self) -> Dict[str, int]:
        """返回特征维度信息。"""
        return {
            'node_dim': self.atom_feature_dim,
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
