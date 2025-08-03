import os
import torch
import argparse
import json
import pickle
import sys
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'model'))

from model.phys_esa import PhysESA
from experiments.stage2.train_stage2 import MultiScaleCollater, Stage2Dataset
from molecular_graph import MultiScaleGraphBuilder

def visualize_top_interactions_3d(
    model: PhysESA,
    data_sample: Data,
    output_path: str,
    collater: MultiScaleCollater,
    top_k_percent: float = 0.1
):
    """
    可视化单个样本中注意力分数最高的粗粒度相互作用网络。
    """
    model.eval()
    
    batch = collater([data_sample])
    device = next(model.parameters()).device
    batch = batch.to(device)

    with torch.no_grad():
        predictions, attention_weights = model(batch, return_attention=True)

    print(f"模型预测亲和力: {predictions.item():.4f}")

    if 'decoder_layer_0' not in attention_weights:
        print("错误: 未找到 'decoder_layer_0' (PMA) 的注意力权重。")
        return

    pma_attention = attention_weights['decoder_layer_0'].squeeze(0)
    
    # 1. 聚合每个边的注意力分数 (跨头和种子向量)
    # Shape: (num_heads, num_seeds, num_edges) -> (num_edges,)
    edge_scores = pma_attention.mean(dim=[0, 1])
    
    # 裁剪掉填充部分
    num_real_edges = data_sample.coarse_edge_index.shape[1]
    edge_scores = edge_scores[:num_real_edges]

    # 2. 筛选出注意力最高的边
    num_top_edges = int(num_real_edges * top_k_percent)
    if num_top_edges == 0 and num_real_edges > 0:
        num_top_edges = 1 # 至少保留一条边
        
    top_scores, top_indices = torch.topk(edge_scores, k=num_top_edges)
    top_edges = data_sample.coarse_edge_index[:, top_indices.cpu()]

    print(f"总粗粒度边数: {num_real_edges}")
    print(f"保留Top {top_k_percent*100:.0f}% 的边: {num_top_edges}")

    # 3. 使用 NetworkX 创建和绘制图形
    G = nx.Graph()
    
    # 添加边
    edges_to_add = top_edges.cpu().numpy().T
    G.add_edges_from(edges_to_add)

    # 绘制图形
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42) # 使用弹簧布局
    
    nx.draw(G, pos, with_labels=False, node_color='skyblue', node_size=80, edge_color='gray', width=1.0, alpha=0.7)
    
    plt.title(f"Top {top_k_percent*100:.0f}% Interaction Network for {getattr(data_sample, 'complex_id', 'Unknown')}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"2D网络图已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PhysESA Top Interaction Network Visualization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.ckpt).")
    # --- 模式一: 从数据集中加载 ---
    parser.add_argument("--data_dir", type=str, default="./experiments/stage2", help="Directory containing the processed data (for loading from dataset).")
    parser.add_argument("--sample_id", type=str, default=None, help="Specific complex ID to visualize from the test set (e.g., '1a4k').")
    
    # --- 模式二: 从原始文件预测 ---
    parser.add_argument("--pdb_file", type=str, default=None, help="Path to a new protein PDB file.")
    parser.add_argument("--mol_file", type=str, default=None, help="Path to a new ligand MOL file.")
    parser.add_argument("--complex_id", type=str, default="new_complex", help="Name for the new complex.")
    parser.add_argument("--output", type=str, default="top_interactions.png", help="Path to save the output PNG image file.")
    parser.add_argument("--top_k", type=float, default=0.1, help="Percentage of top edges to keep (e.g., 0.1 for 10%).")
    
    args = parser.parse_args()

    model = PhysESA.load_from_checkpoint(args.checkpoint)
    print("模型已加载。")

    # --- 根据参数决定执行模式 ---
    if args.pdb_file and args.mol_file:
        # 模式二: 从原始文件动态构建图
        print("--- 模式: 从文件预测新复合物 ---")
        print(f"蛋白质 PDB: {args.pdb_file}")
        print(f"配体 MOL: {args.mol_file}")

        # 使用默认配置创建GraphBuilder
        # 注意：这里的配置需要与训练时大致匹配
        graph_builder = MultiScaleGraphBuilder(
            interface_cutoff=8.0,
            use_knn=True,
            k=16
        )
        
        data_sample = graph_builder.build_graph(
            complex_id=args.complex_id,
            pdb_file=args.pdb_file,
            sdf_file=args.mol_file
        )
        
        if data_sample is None:
            raise ValueError("无法从提供的PDB/SDF文件构建图。请检查文件格式或内容。")
        
        print(f"成功为 '{args.complex_id}' 构建图。")
        
        # 对于单个动态图，我们需要一个能处理它的Collater
        # 我们可以基于这个图的尺寸动态创建一个Collater
        collater = MultiScaleCollater(
            atomic_max_nodes=data_sample.num_nodes,
            atomic_max_edges=data_sample.edge_index.shape[1],
            coarse_max_nodes=data_sample.num_coarse_nodes,
            coarse_max_edges=data_sample.coarse_edge_index.shape[1]
        )

    else:
        # 模式一: 从预处理的数据集中加载
        print("--- 模式: 从数据集中加载样本 ---")
        metadata_path = os.path.join(args.data_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        padding_config = metadata.get('padding_dimensions')
        if not padding_config:
            raise ValueError("错误: metadata.json 中未找到 'padding_dimensions'。")

        collater = MultiScaleCollater(
            atomic_max_nodes=padding_config['atomic_nodes'],
            atomic_max_edges=padding_config['atomic_edges'],
            coarse_max_nodes=padding_config['coarse_nodes'],
            coarse_max_edges=padding_config['coarse_edges']
        )

        test_dataset = Stage2Dataset(os.path.join(args.data_dir, 'test.pkl'))
        
        if args.sample_id:
            sample_idx = next((i for i, data in enumerate(test_dataset.data) if data.complex_id == args.sample_id), -1)
            if sample_idx == -1:
                raise ValueError(f"Sample ID {args.sample_id} not found in the test set.")
            data_sample = test_dataset[sample_idx]
        else:
            print("未指定样本ID，使用测试集中的第一个样本。")
            data_sample = test_dataset[0]

    print(f"正在可视化样本: {getattr(data_sample, 'complex_id', 'Unknown')}")

    visualize_top_interactions_3d(model, data_sample, args.output, collater, top_k_percent=args.top_k)

if __name__ == "__main__":
    main()