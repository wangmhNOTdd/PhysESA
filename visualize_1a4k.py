#!/usr/bin/env python3
"""
蛋白质-小分子复合物1a4k的3D可视化
展示原子在3D空间中的分布
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from Bio import PDB
from rdkit import Chem
import os
import json

# 定义原子颜色方案 (CPK颜色)
ATOM_COLORS = {
    'H': '#FFFFFF',   # 白色
    'C': '#909090',   # 灰色
    'N': '#3050F8',   # 蓝色
    'O': '#FF0D0D',   # 红色
    'F': '#90E050',   # 绿色
    'P': '#FF8000',   # 橙色
    'S': '#FFFF30',   # 黄色
    'CL': '#1FF01F',  # 绿色
    'BR': '#A62929',  # 棕色
    'I': '#940094',   # 紫色
    'FE': '#E06633',  # 铁锈色
    'CA': '#3DFF00',  # 亮绿色
}

# Van der Waals半径 (埃)
VDW_RADII = {
    'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47,
    'P': 1.8, 'S': 1.8, 'CL': 1.75, 'BR': 1.85, 'I': 1.98,
    'FE': 2.0, 'CA': 2.0
}

def parse_pdb_file(pdb_file):
    """解析PDB文件，提取原子坐标信息"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    atoms_data = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coord = atom.get_coord()
                    atoms_data.append({
                        'chain_id': chain.id,
                        'residue_name': residue.get_resname(),
                        'residue_id': residue.get_id()[1],
                        'atom_name': atom.get_name(),
                        'element': atom.element,
                        'x': coord[0],
                        'y': coord[1],
                        'z': coord[2],
                        'bfactor': atom.get_bfactor(),
                        'occupancy': atom.get_occupancy()
                    })
    
    return pd.DataFrame(atoms_data)

def parse_sdf_ligand(sdf_file):
    """解析SDF文件，提取配体原子坐标"""
    mol = Chem.MolFromMolFile(sdf_file)
    if mol is None:
        return None, None
    
    # 获取3D坐标
    conf = mol.GetConformer()
    atoms_data = []
    
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms_data.append({
            'atom_idx': atom.GetIdx(),
            'element': atom.GetSymbol(),
            'x': pos.x,
            'y': pos.y,
            'z': pos.z,
            'formal_charge': atom.GetFormalCharge(),
            'hybridization': str(atom.GetHybridization()),
        })
    
    return pd.DataFrame(atoms_data), mol

def create_3d_visualization(protein_df, ligand_df, complex_name="1a4k"):
    """创建3D可视化图表"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'{complex_name} - 完整复合物',
            f'{complex_name} - 蛋白质结构',
            f'{complex_name} - 配体结构',
            f'{complex_name} - 原子密度分布'
        ),
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}],
               [{"type": "scatter3d"}, {"type": "histogram"}]],
        vertical_spacing=0.1
    )
    
    # 1. 完整复合物视图
    if not protein_df.empty:
        # 蛋白质原子
        for element in protein_df['element'].unique():
            if pd.isna(element) or element == '':
                continue
            element_data = protein_df[protein_df['element'] == element]
            
            fig.add_trace(
                go.Scatter3d(
                    x=element_data['x'],
                    y=element_data['y'],
                    z=element_data['z'],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=ATOM_COLORS.get(element, '#808080'),
                        opacity=0.6
                    ),
                    name=f'蛋白质-{element}',
                    text=[f'链: {chain}<br>残基: {res}<br>原子: {atom}<br>坐标: ({x:.2f}, {y:.2f}, {z:.2f})' 
                          for chain, res, atom, x, y, z in 
                          zip(element_data['chain_id'], element_data['residue_name'], 
                              element_data['atom_name'], element_data['x'], 
                              element_data['y'], element_data['z'])],
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=1, col=1
            )
    
    if not ligand_df.empty:
        # 配体原子
        for element in ligand_df['element'].unique():
            element_data = ligand_df[ligand_df['element'] == element]
            
            fig.add_trace(
                go.Scatter3d(
                    x=element_data['x'],
                    y=element_data['y'],
                    z=element_data['z'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=ATOM_COLORS.get(element, '#808080'),
                        opacity=0.9,
                        line=dict(width=2, color='black')
                    ),
                    name=f'配体-{element}',
                    text=[f'配体原子: {element}<br>索引: {idx}<br>坐标: ({x:.2f}, {y:.2f}, {z:.2f})' 
                          for idx, x, y, z in 
                          zip(element_data['atom_idx'], element_data['x'], 
                              element_data['y'], element_data['z'])],
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=1, col=1
            )
    
    # 2. 蛋白质结构视图
    if not protein_df.empty:
        # 按链着色
        chain_colors = px.colors.qualitative.Set3
        unique_chains = protein_df['chain_id'].unique()
        
        for i, chain in enumerate(unique_chains):
            chain_data = protein_df[protein_df['chain_id'] == chain]
            
            fig.add_trace(
                go.Scatter3d(
                    x=chain_data['x'],
                    y=chain_data['y'],
                    z=chain_data['z'],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=chain_colors[i % len(chain_colors)],
                        opacity=0.7
                    ),
                    name=f'链 {chain}',
                    text=[f'链: {chain}<br>残基: {res}<br>原子: {atom}' 
                          for chain, res, atom in 
                          zip(chain_data['chain_id'], chain_data['residue_name'], 
                              chain_data['atom_name'])],
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=1, col=2
            )
    
    # 3. 配体结构视图
    if not ligand_df.empty:
        for element in ligand_df['element'].unique():
            element_data = ligand_df[ligand_df['element'] == element]
            
            fig.add_trace(
                go.Scatter3d(
                    x=element_data['x'],
                    y=element_data['y'],
                    z=element_data['z'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=ATOM_COLORS.get(element, '#808080'),
                        opacity=0.9,
                        line=dict(width=2, color='black')
                    ),
                    name=f'配体-{element}',
                    text=[f'原子: {element}<br>索引: {idx}<br>坐标: ({x:.2f}, {y:.2f}, {z:.2f})' 
                          for idx, x, y, z in 
                          zip(element_data['atom_idx'], element_data['x'], 
                              element_data['y'], element_data['z'])],
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=2, col=1
            )
    
    # 4. 原子密度分布
    all_atoms = []
    if not protein_df.empty:
        all_atoms.extend(protein_df['element'].tolist())
    if not ligand_df.empty:
        all_atoms.extend(ligand_df['element'].tolist())
    
    if all_atoms:
        element_counts = pd.Series(all_atoms).value_counts()
        
        fig.add_trace(
            go.Bar(
                x=element_counts.index,
                y=element_counts.values,
                marker_color=[ATOM_COLORS.get(elem, '#808080') for elem in element_counts.index],
                name='原子计数',
                text=element_counts.values,
                textposition='outside'
            ),
            row=2, col=2
        )
    
    # 更新布局
    fig.update_layout(
        title=f'PDBbind 复合物 {complex_name} - 3D原子分布可视化',
        height=1000,
        showlegend=True,
        font=dict(size=10)
    )
    
    # 更新3D场景
    scene_layout = dict(
        xaxis_title='X (Å)',
        yaxis_title='Y (Å)',
        zaxis_title='Z (Å)',
        aspectmode='cube'
    )
    
    fig.update_scenes(scene_layout)
    
    return fig

def load_metadata_info(complex_id):
    """加载元数据信息"""
    metadata_path = "./datasets/pdbbind/metadata"
    info = {}
    
    try:
        # 亲和力数据
        with open(f"{metadata_path}/affinities.json", 'r') as f:
            affinities = json.load(f)
            info['affinity'] = affinities.get(complex_id, 'Unknown')
        
        # SMILES数据
        with open(f"{metadata_path}/lig_smiles.json", 'r') as f:
            smiles = json.load(f)
            info['smiles'] = smiles.get(complex_id, 'Unknown')
    except:
        info['affinity'] = 'Unknown'
        info['smiles'] = 'Unknown'
    
    return info

def main():
    """主函数"""
    print("=== PDBbind 1a28 复合物3D可视化分析 ===\n")
    
    # 数据文件路径 - 更换为1a28复合物
    complex_id = "1a28"
    pdb_file = f"./datasets/pdbbind/pdb_files/{complex_id}/{complex_id}.pdb"
    sdf_file = f"./datasets/pdbbind/pdb_files/{complex_id}/{complex_id}_ligand.sdf"
    
    # 检查文件存在性
    if not os.path.exists(pdb_file):
        print(f"错误: 找不到PDB文件 {pdb_file}")
        return
    if not os.path.exists(sdf_file):
        print(f"错误: 找不到SDF文件 {sdf_file}")
        return
    
    print("正在解析文件...")
    
    # 解析蛋白质结构
    try:
        protein_df = parse_pdb_file(pdb_file)
        print(f"✓ 蛋白质原子数量: {len(protein_df)}")
        print(f"  - 链数量: {protein_df['chain_id'].nunique()}")
        print(f"  - 残基数量: {protein_df['residue_id'].nunique()}")
        print(f"  - 元素类型: {', '.join(protein_df['element'].unique())}")
    except Exception as e:
        print(f"错误: 解析PDB文件失败 - {e}")
        protein_df = pd.DataFrame()
    
    # 解析配体结构
    try:
        ligand_df, mol = parse_sdf_ligand(sdf_file)
        if ligand_df is not None:
            print(f"✓ 配体原子数量: {len(ligand_df)}")
            print(f"  - 元素类型: {', '.join(ligand_df['element'].unique())}")
        else:
            print("✗ 配体解析失败")
            ligand_df = pd.DataFrame()
    except Exception as e:
        print(f"错误: 解析SDF文件失败 - {e}")
        ligand_df = pd.DataFrame()
        mol = None
    
    # 加载元数据
    metadata = load_metadata_info(complex_id)
    print(f"\n=== 复合物信息 ===")
    print(f"PDB ID: {complex_id}")
    print(f"结合亲和力: {metadata['affinity']} (pKd)")
    print(f"配体SMILES: {metadata['smiles'][:50]}..." if len(str(metadata['smiles'])) > 50 else f"配体SMILES: {metadata['smiles']}")
    
    # 计算一些统计信息
    if not protein_df.empty and not ligand_df.empty:
        # 计算结合口袋中心
        protein_center = [protein_df['x'].mean(), protein_df['y'].mean(), protein_df['z'].mean()]
        ligand_center = [ligand_df['x'].mean(), ligand_df['y'].mean(), ligand_df['z'].mean()]
        distance = np.sqrt(sum((p-l)**2 for p, l in zip(protein_center, ligand_center)))
        
        print(f"\n=== 空间分析 ===")
        print(f"蛋白质中心: ({protein_center[0]:.2f}, {protein_center[1]:.2f}, {protein_center[2]:.2f})")
        print(f"配体中心: ({ligand_center[0]:.2f}, {ligand_center[1]:.2f}, {ligand_center[2]:.2f})")
        print(f"蛋白质-配体中心距离: {distance:.2f} Å")
    
    # 创建可视化
    print("\n正在生成3D可视化...")
    fig = create_3d_visualization(protein_df, ligand_df, complex_id)
    
    # 保存图表
    output_file = f"{complex_id}_3d_visualization.html"
    fig.write_html(output_file)
    print(f"✓ 3D可视化已保存到: {output_file}")
    
    # 显示图表
    fig.show()
    
    print("\n=== 分析完成 ===")
    print("可视化图表已在浏览器中打开，展示了:")
    print("1. 完整复合物的3D原子分布")
    print("2. 蛋白质结构（按链着色）")
    print("3. 配体结构（按元素着色）")
    print("4. 原子元素统计分布")

if __name__ == "__main__":
    main()
