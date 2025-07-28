"""
阶段二训练脚本
使用PhysESA模型进行蛋白质-小分子亲和力预测
"""

import os
import pickle
import json
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import math
from typing import List, Dict, Any
import argparse
import sys
from torch_geometric.data import Batch

# 添加项目根目录和model目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'model'))

from model.phys_esa import PhysESA

class Stage2Dataset(Dataset):
    """阶段二数据集类"""
    def __init__(self, data_path: str):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"加载数据集: {data_path}, 样本数: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MultiScaleCollater:
    """
    为多尺度图数据定制的Collater。
    它负责正确批处理包含原子和粗粒度图信息的Data对象。
    """
    def __init__(self, atomic_max_nodes, atomic_max_edges, coarse_max_nodes, coarse_max_edges):
        self.atomic_max_nodes = atomic_max_nodes
        self.atomic_max_edges = atomic_max_edges
        self.coarse_max_nodes = coarse_max_nodes
        self.coarse_max_edges = coarse_max_edges

    def __call__(self, batch: List[Any]) -> Batch:
        if not batch:
            return None # 如果批次为空，直接返回

        # 1. PyG的默认批处理会正确处理 `edge_index` 和 `batch` 属性
        batch_data = Batch.from_data_list(batch)

        # 2. 手动处理自定义图属性的批次偏移
        coarse_node_offset = 0
        new_coarse_edge_indices = []
        
        for i, data in enumerate(batch):
            # 偏移 atom_to_coarse_idx
            batch_data.atom_to_coarse_idx[batch_data.ptr[i]:batch_data.ptr[i+1]] += coarse_node_offset
            
            # 手动偏移并收集 coarse_edge_index
            if data.coarse_edge_index.numel() > 0:
                new_coarse_edge_indices.append(data.coarse_edge_index + coarse_node_offset)
            
            coarse_node_offset += data.num_coarse_nodes

        # 拼接所有偏移后的 coarse_edge_index
        if new_coarse_edge_indices:
            batch_data.coarse_edge_index = torch.cat(new_coarse_edge_indices, dim=1)
        else:
            # 如果整个批次都没有粗粒度边
            batch_data.coarse_edge_index = torch.empty((2, 0), dtype=torch.long)

        # 3. 为粗粒度图创建batch和ptr (现在可以安全地创建了)
        coarse_num_nodes_list = [data.num_coarse_nodes for data in batch]
        coarse_batch_list = [torch.full((num,), i, dtype=torch.long) for i, num in enumerate(coarse_num_nodes_list)]
        batch_data.coarse_batch = torch.cat(coarse_batch_list, dim=0)
        batch_data.coarse_ptr = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(coarse_num_nodes_list), 0)], dim=0)

        # 4. 将最大尺寸附加到批处理对象中
        batch_data.max_node_global = torch.tensor([self.atomic_max_nodes], dtype=torch.long)
        batch_data.max_edge_global = torch.tensor([self.atomic_max_edges], dtype=torch.long)
        batch_data.coarse_max_node_global = torch.tensor([self.coarse_max_nodes], dtype=torch.long)
        batch_data.coarse_max_edge_global = torch.tensor([self.coarse_max_edges], dtype=torch.long)
        
        # 附加总的粗粒度节点数，用于scatter的dim_size
        batch_data.num_coarse_nodes = coarse_node_offset

        return batch_data

def main():
    parser = argparse.ArgumentParser(description='阶段二PhysESA模型训练')
    parser.add_argument('--data_dir', type=str, default='./experiments/stage2', help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./experiments/stage2', help='输出和检查点目录')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径（可选）')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径，用于测试模式')
    # ... (可以添加更多命令行参数来覆盖配置)
    args = parser.parse_args()

    # 性能优化建议
    torch.set_float32_matmul_precision('medium')

    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- 1. 加载配置 ---
    metadata_path = os.path.join(args.data_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"元数据文件未找到: {metadata_path}。请先运行prepare_stage2_data.py。")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    feature_dims = metadata['feature_dimensions']
    
    # 默认训练配置
    training_config = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'lr_patience': 10,
        'batch_size': 8,
        'max_epochs': 40,
        'patience': 15,
        'grad_clip': 1.0,
        'use_fp16': True,
        'num_workers': 8
    }
    
    # 默认ESA模型配置
    esa_config = {
        # --- Estimator's MLP parameters ---
        'graph_dim': 128,
        # --- ESA model parameters ---
        'hidden_dims': [128, 128, 128, 128, 128, 128, 128, 128],
        'num_heads': [8, 8, 8, 8, 8, 8, 8, 8],
        'layer_types': ['M', 'M', 'S', 'M', 'M', 'S', 'P', 'S'],
        'num_inds': 32,
        'set_max_items': 0, # This will be set dynamically later
        'linear_output_size': 1,
        'use_fp16': True, # Corresponds to training_config
        'node_or_edge': "edge", # Renamed from apply_attention_on
        'xformers_or_torch_attn': "xformers",
        'pre_or_post': "pre",
        'norm_type': "LN",
        'sab_dropout': 0.1,
        'mab_dropout': 0.1,
        'pma_dropout': 0.1,
        'residual_dropout': 0.0,
        'pma_residual_dropout': 0.0,
        'use_mlps': True,
        'mlp_hidden_size': 128, # A sensible default
        'num_mlp_layers': 2,
        'mlp_type': "gated_mlp",
        'mlp_dropout': 0.1, # A sensible default
        'use_mlp_ln': False,
    }

    # --- 2. 准备数据加载器 ---
    train_dataset = Stage2Dataset(os.path.join(args.data_dir, 'train.pkl'))
    val_dataset = Stage2Dataset(os.path.join(args.data_dir, 'valid.pkl'))
    test_dataset = Stage2Dataset(os.path.join(args.data_dir, 'test.pkl'))
    
    # --- 为多尺度图计算全局最大节点/边数 ---
    full_dataset = train_dataset.data + val_dataset.data + test_dataset.data
    
    # 原子图
    atomic_node_counts = [d.num_nodes for d in full_dataset]
    atomic_edge_counts = [d.edge_index.shape[1] for d in full_dataset]
    
    # 粗粒度图
    coarse_node_counts = [d.num_coarse_nodes for d in full_dataset]
    coarse_edge_counts = [d.coarse_edge_index.shape[1] for d in full_dataset]

    def nearest_multiple_of_8(n):
        return math.ceil(n / 8) * 8

    # 计算原子图的填充尺寸
    raw_max_atomic_nodes = max(atomic_node_counts) if atomic_node_counts else 0
    raw_max_atomic_edges = max(atomic_edge_counts) if atomic_edge_counts else 0
    final_max_atomic_nodes = nearest_multiple_of_8(raw_max_atomic_nodes + 1)
    final_max_atomic_edges = nearest_multiple_of_8(raw_max_atomic_edges + 1)

    # 计算粗粒度图的填充尺寸
    raw_max_coarse_nodes = max(coarse_node_counts) if coarse_node_counts else 0
    raw_max_coarse_edges = max(coarse_edge_counts) if coarse_edge_counts else 0
    final_max_coarse_nodes = nearest_multiple_of_8(raw_max_coarse_nodes + 1)
    final_max_coarse_edges = nearest_multiple_of_8(raw_max_coarse_edges + 1)

    # 为两个尺度的模型设置填充大小
    esa_config['atomic_set_max_items'] = raw_max_atomic_edges
    esa_config['coarse_set_max_items'] = raw_max_coarse_edges
    
    collater = MultiScaleCollater(
        atomic_max_nodes=final_max_atomic_nodes,
        atomic_max_edges=final_max_atomic_edges,
        coarse_max_nodes=final_max_coarse_nodes,
        coarse_max_edges=final_max_coarse_edges
    )
    
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True, num_workers=training_config['num_workers'], collate_fn=collater)
    val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], shuffle=False, num_workers=training_config['num_workers'], collate_fn=collater)
    test_loader = DataLoader(test_dataset, batch_size=training_config['batch_size'], shuffle=False, num_workers=training_config['num_workers'], collate_fn=collater)

    # --- 3. 打印最终配置并初始化模型 ---
    print("\n" + "="*50)
    print(" " * 15 + "FINAL CONFIGURATIONS")
    print("="*50)
    print("\n[Training Configuration]")
    for key, value in training_config.items():
        print(f"  - {key:<20}: {value}")
    
    print("\n[ESA Configuration]")
    for key, value in esa_config.items():
        # 避免打印过长的列表
        if isinstance(value, list) and len(value) > 5:
            print(f"  - {key:<20}: [{value[0]}, {value[1]}, ... , {value[-1]}] (len={len(value)})")
        else:
            print(f"  - {key:<20}: {value}")
    print("="*50 + "\n")

    model = PhysESA(
        esa_config=esa_config,
        feature_dims=feature_dims,
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )

    # --- 4. 设置训练器 ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='physesa-{epoch:02d}-{val_loss:.3f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=training_config['patience'],
        mode='min',
        verbose=True
    )
    logger = TensorBoardLogger(save_dir=args.output_dir, name='physesa_logs')

    trainer = pl.Trainer(
        max_epochs=training_config['max_epochs'],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision="bf16" if training_config['use_fp16'] else 32,
        gradient_clip_val=training_config['grad_clip']
    )

    # --- 5. 开始训练或测试 ---
    if args.checkpoint:
        print(f"开始测试模型: {args.checkpoint}")
        # 加载模型时，需要传递原始的配置参数，因为它们保存在hyperparameters中
        model_for_test = PhysESA.load_from_checkpoint(
            args.checkpoint,
            esa_config=esa_config,
            training_config=training_config,
            feature_dims=feature_dims
        )
        trainer.test(model_for_test, dataloaders=test_loader)
    else:
        print("开始训练PhysESA模型...")
        trainer.fit(model, train_loader, val_loader)
        print(f"训练完成！模型和日志保存在: {args.output_dir}")
        
        # 增加一个短暂延时，确保logger有时间写入文件
        import time
        print("等待日志文件写入...")
        time.sleep(2)
        
        # 在训练结束后，使用最佳模型进行测试
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            print(f"\n使用最佳模型进行测试: {best_model_path}")
            trainer.test(model, dataloaders=test_loader, ckpt_path=best_model_path)
        else:
            print(f"\n警告: 未找到最佳模型检查点 '{best_model_path}'，跳过最终测试。")

    return args

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_results(log_dir: str):
    """
    从TensorBoard日志中读取metrics.csv并绘制训练曲线。
    """
    metrics_path = os.path.join(log_dir, 'metrics.csv')
    if not os.path.exists(metrics_path):
        print(f"警告: 在 {log_dir} 中未找到 metrics.csv，跳过绘图。")
        return

    print(f"从 {metrics_path} 加载指标并开始绘图...")
    metrics_df = pd.read_csv(metrics_path)

    # 数据预处理：处理每个epoch可能有多行（train_loss_step）的情况
    # 我们只关心每个epoch结束时的指标
    epoch_metrics = metrics_df.groupby('epoch').last().reset_index()

    # 获取所有记录的指标
    val_metrics = sorted([col for col in epoch_metrics.columns if col.startswith('val/')])
    
    # 设置绘图风格
    sns.set_style("whitegrid")
    
    # 计算需要的行数
    num_plots = 1 + len(val_metrics)
    num_rows = (num_plots + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(20, 6 * num_rows), squeeze=False)
    axes = axes.flatten()

    # 1. 绘制训练和验证损失
    ax = axes[0]
    if 'train_loss_epoch' in epoch_metrics.columns:
        sns.lineplot(data=epoch_metrics, x='epoch', y='train_loss_epoch', ax=ax, label='Train Loss (epoch)', color='royalblue', marker='o', markersize=4)
    if 'val_loss' in epoch_metrics.columns:
        sns.lineplot(data=epoch_metrics, x='epoch', y='val_loss', ax=ax, label='Validation Loss', color='darkorange', marker='o', markersize=4)
    ax.set_title('Training & Validation Loss vs. Epoch', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend()
    
    # 找到并标注最佳epoch
    # 找到并标注最佳epoch
    if 'val_loss' in epoch_metrics.columns and not epoch_metrics['val_loss'].isnull().all():
        best_epoch_row = epoch_metrics.loc[epoch_metrics['val_loss'].idxmin()]
        best_epoch = int(best_epoch_row['epoch'])
        best_val_loss = float(best_epoch_row['val_loss'])
        
        # 标注损失图
        ax.axvline(x=best_epoch, color='red', linestyle='--', linewidth=1.5, label=f'Best Epoch: {best_epoch}')
        ax.scatter(x=best_epoch, y=best_val_loss, marker='*', s=200, color='red', zorder=5, label=f'Best val_loss: {best_val_loss:.4f}')
        ax.legend()

        # 2. 绘制其他验证集指标
        plot_idx = 1
        for metric in val_metrics:
            if metric == 'val_loss': continue
            ax = axes[plot_idx]
            sns.lineplot(data=epoch_metrics, x='epoch', y=metric, ax=ax, label=metric, marker='o', markersize=4)
            ax.set_title(f'Validation {metric.split("/")[-1].capitalize()} vs. Epoch', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            
            # 在其他指标图上也标注最佳epoch的位置
            best_metric_val = float(best_epoch_row[metric])
            ax.axvline(x=best_epoch, color='red', linestyle='--', linewidth=1.5)
            ax.scatter(x=best_epoch, y=best_metric_val, marker='*', s=200, color='red', zorder=5, label=f'Value at Best Epoch: {best_metric_val:.4f}')
            ax.legend()
            plot_idx += 1
    else:
        # 如果没有val_loss，正常绘制其他图但不加标注
        plot_idx = 1
        for metric in val_metrics:
            if metric == 'val_loss': continue
            ax = axes[plot_idx]
            sns.lineplot(data=epoch_metrics, x='epoch', y=metric, ax=ax, label=metric, marker='o', markersize=4)
            ax.set_title(f'Validation {metric.split("/")[-1].capitalize()} vs. Epoch', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.legend()
            plot_idx += 1

    # 隐藏多余的子图
    for i in range(plot_idx, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle('Training and Validation Metrics', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path = os.path.join(os.path.dirname(log_dir), 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线图已保存至: {save_path}")
    plt.close()


if __name__ == "__main__":
    args = main()
    if args:
        # 训练/测试结束后，找到最新的日志目录并绘图
        log_dir = os.path.join(args.output_dir, 'physesa_logs')
        if os.path.exists(log_dir):
            versions = sorted([d for d in os.listdir(log_dir) if d.startswith('version_')])
            if versions:
                latest_version_dir = os.path.join(log_dir, versions[-1])
                plot_training_results(latest_version_dir)