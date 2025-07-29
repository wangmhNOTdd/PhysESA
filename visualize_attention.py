import os
import torch
import argparse
import json
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'model'))

from model.phys_esa import PhysESA
from experiments.stage2.train_stage2 import MultiScaleCollater, Stage2Dataset

def get_coarse_graph_edge_labels(data):
    """为粗粒度图的边生成可读的标签。"""
    # 这是一个简化的示例，需要根据您的具体数据结构进行调整
    # 我们需要一个方法来从 coarse_node_idx 映射回 res_id 或 motif_id
    # 这部分信息目前不在Data对象中，需要从原始数据构建过程中获取
    # 作为临时方案，我们只使用索引
    
    num_coarse_nodes = data.num_coarse_nodes
    node_labels = [f"Node_{i}" for i in range(num_coarse_nodes)] # 简化标签

    edge_labels = []
    for i in range(data.coarse_edge_index.shape[1]):
        src_idx = data.coarse_edge_index[0, i].item()
        dst_idx = data.coarse_edge_index[1, i].item()
        edge_labels.append(f"{node_labels[src_idx]}-{node_labels[dst_idx]}")
    return edge_labels

def visualize_attention(
    model: PhysESA,
    data_sample: torch.utils.data.Dataset,
    output_path: str,
    collater: MultiScaleCollater
):
    """
    可视化单个样本的粗粒度图注意力权重。
    """
    model.eval()
    
    # 使用 collater 来创建批次，即使只有一个样本
    batch = collater([data_sample])
    
    # 将数据移动到模型所在的设备
    device = next(model.parameters()).device
    batch = batch.to(device)

    with torch.no_grad():
        # 请求返回注意力权重
        predictions, attention_weights = model(batch, return_attention=True)

    print(f"模型预测亲和力: {predictions.item():.4f}")
    print("提取到的注意力权重字典键:", attention_weights.keys())

    # 我们最关心PMA层的注意力，它在粗粒度编码器的解码器第一层
    # 根据ESA的结构，这通常是 'decoder_layer_0'
    if 'decoder_layer_0' not in attention_weights:
        print("错误: 未找到 'decoder_layer_0' 的注意力权重。请检查模型结构。")
        return

    # (batch_size, num_heads, num_seeds, num_edges)
    pma_attention = attention_weights['decoder_layer_0'].squeeze(0) # 移除batch维度
    
    # 为了可视化，我们将所有头的注意力平均起来
    # (num_seeds, num_edges)
    avg_pma_attention = pma_attention.mean(dim=0) 

    # 获取边的标签
    edge_labels = get_coarse_graph_edge_labels(data_sample)
    
    # 检查维度是否匹配
    if avg_pma_attention.shape[1] != len(edge_labels):
        print(f"警告: 注意力权重维度 ({avg_pma_attention.shape[1]}) 与边标签数量 ({len(edge_labels)}) 不匹配。")
        # 可能是由于填充（padding）导致的，我们只取有效部分
        avg_pma_attention = avg_pma_attention[:, :len(edge_labels)]

    # 创建DataFrame以便于绘图
    df = pd.DataFrame(avg_pma_attention.cpu().numpy(), columns=edge_labels)
    df.index = [f"Seed_{i}" for i in range(df.shape[0])]

    # 绘图
    plt.figure(figsize=(max(20, len(edge_labels) // 2), max(10, df.shape[0] // 2)))
    sns.heatmap(df, cmap="viridis", annot=False) # 对于大型矩阵，关闭annot
    plt.title(f"PMA Attention Weights for {data_sample.complex_id}", fontsize=16)
    plt.xlabel("Coarse-Grained Edges (Residue/Motif Interactions)", fontsize=12)
    plt.ylabel("PMA Seed Vectors", fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"注意力热力图已保存至: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="PhysESA Attention Visualization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.ckpt).")
    parser.add_argument("--data_dir", type=str, default="./experiments/stage2", help="Directory containing the processed data.")
    parser.add_argument("--sample_id", type=str, default=None, help="Specific complex ID to visualize (e.g., '1a4k'). If not provided, the first sample from the test set is used.")
    parser.add_argument("--output", type=str, default="attention_heatmap.png", help="Path to save the output heatmap image.")
    
    args = parser.parse_args()

    # --- 1. 加载配置和数据 ---
    # 我们需要从训练脚本中获取配置信息
    metadata_path = os.path.join(args.data_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # 加载模型时，PyTorch Lightning会自动加载hyperparameters
    # 我们只需要确保模型类可被找到
    model = PhysESA.load_from_checkpoint(args.checkpoint)
    print("模型已加载。")

    # --- 2. 准备数据加载器和Collater ---
    # 我们需要collater来正确构建批次
    # 注意：这里的max_nodes/edges需要和训练时一致，我们从模型的超参数中获取
    atomic_max_edges = model.hparams.esa_config['atomic_encoder_config']['set_max_items']
    coarse_max_edges = model.hparams.esa_config['coarse_encoder_config']['set_max_items']
    
    # 注意：max_nodes没有直接保存在esa_config中，需要一个更好的方式来获取
    # 暂时使用一个估算值，但这可能不是最优的
    print("警告: max_nodes/edges 的恢复依赖于超参数，请确保检查点包含正确的配置。")
    
    collater = MultiScaleCollater(
        atomic_max_nodes=atomic_max_edges * 2, # 粗略估计
        atomic_max_edges=atomic_max_edges + 8,
        coarse_max_nodes=coarse_max_edges * 2, # 粗略估计
        coarse_max_edges=coarse_max_edges + 8
    )

    # --- 3. 加载特定样本 ---
    test_dataset = Stage2Dataset(os.path.join(args.data_dir, 'test.pkl'))
    
    if args.sample_id:
        sample_idx = -1
        for i, data in enumerate(test_dataset.data):
            if data.complex_id == args.sample_id:
                sample_idx = i
                break
        if sample_idx == -1:
            raise ValueError(f"Sample ID {args.sample_id} not found in the test set.")
        data_sample = test_dataset[sample_idx]
    else:
        print("未指定样本ID，使用测试集中的第一个样本。")
        data_sample = test_dataset[0]

    print(f"正在可视化样本: {data_sample.complex_id}")

    # --- 4. 运行可视化 ---
    visualize_attention(model, data_sample, args.output, collater)

if __name__ == "__main__":
    main()