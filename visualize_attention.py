import os
import torch
import argparse
import json
import pickle
import sys
import py3Dmol
from torch_geometric.data import Data

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'model'))

from model.phys_esa import PhysESA
from experiments.stage2.train_stage2 import MultiScaleCollater, Stage2Dataset

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
    top_edges = data_sample.coarse_edge_index[:, top_indices]

    print(f"总粗粒度边数: {num_real_edges}")
    print(f"保留Top {top_k_percent*100:.0f}% 的边: {num_top_edges}")

    # 3. 准备3D可视化数据
    node_coords = data_sample.coarse_pos.cpu().numpy()
    node_id_map = data_sample.coarse_node_id_map
    
    # 找出所有参与重要相互作用的节点
    involved_nodes = torch.unique(top_edges.flatten()).cpu().numpy()

    # 4. 使用 py3Dmol 创建可视化
    view = py3Dmol.view(width=800, height=600)

    # 添加节点（球体）
    for node_idx in involved_nodes:
        pos = node_coords[node_idx].tolist()
        label = node_id_map.get(node_idx, f"Node_{node_idx}")
        
        if label.startswith("LIG"):
            color = "green"
        else:
            color = "blue"
            
        view.addSphere({
            'center': {'x': pos[0], 'y': pos[1], 'z': pos[2]},
            'radius': 0.8,
            'color': color,
            'alpha': 0.9
        })
        view.addLabel(label, {'position': {'x': pos[0], 'y': pos[1], 'z': pos[2]}, 'fontColor': 'white', 'fontSize': 10, 'backgroundColor': 'black', 'backgroundOpacity': 0.5})

    # 添加边（圆柱）
    for i in range(top_edges.shape[1]):
        src_idx, dst_idx = top_edges[:, i].cpu().numpy()
        src_pos = node_coords[src_idx].tolist()
        dst_pos = node_coords[dst_idx].tolist()
        
        view.addCylinder({
            'start': {'x': src_pos[0], 'y': src_pos[1], 'z': src_pos[2]},
            'end': {'x': dst_pos[0], 'y': dst_pos[1], 'z': dst_pos[2]},
            'color': 'orange',
            'radius': 0.2,
            'dashed': False
        })

    view.zoomTo()
    view.write_html(output_path)
    print(f"3D可视化网络已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PhysESA Top Interaction Network Visualization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.ckpt).")
    parser.add_argument("--data_dir", type=str, default="./experiments/stage2", help="Directory containing the processed data.")
    parser.add_argument("--sample_id", type=str, default=None, help="Specific complex ID to visualize (e.g., '1a4k'). If not provided, the first sample from the test set is used.")
    parser.add_argument("--output", type=str, default="top_interactions.html", help="Path to save the output HTML file.")
    parser.add_argument("--top_k", type=float, default=0.1, help="Percentage of top edges to keep (e.g., 0.1 for 10%).")
    
    args = parser.parse_args()

    model = PhysESA.load_from_checkpoint(args.checkpoint)
    print("模型已加载。")

    metadata_path = os.path.join(args.data_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    padding_config = metadata.get('padding_dimensions')
    if not padding_config:
        raise ValueError("错误: metadata.json 中未找到 'padding_dimensions'。请重新运行 prepare_stage2_data.py。")

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

    print(f"正在可视化样本: {data_sample.complex_id}")

    visualize_top_interactions_3d(model, data_sample, args.output, collater, top_k_percent=args.top_k)

if __name__ == "__main__":
    main()