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

class Collater:
    """将Data对象列表批处理为Batch对象"""
    def __init__(self, global_max_edges: int, global_max_nodes: int):
        self.global_max_edges = global_max_edges
        self.global_max_nodes = global_max_nodes

    def __call__(self, batch: List[Any]) -> Batch:
        batch_data = Batch.from_data_list(batch)
        batch_data.max_node_global = torch.tensor([self.global_max_nodes], dtype=torch.long)
        batch_data.max_edge_global = torch.tensor([self.global_max_edges], dtype=torch.long)
        return batch_data

def main():
    parser = argparse.ArgumentParser(description='阶段二PhysESA模型训练')
    parser.add_argument('--data_dir', type=str, default='./experiments/stage2', help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./experiments/stage2/checkpoints', help='输出目录')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径（可选）')
    # ... (可以添加更多命令行参数来覆盖配置)
    args = parser.parse_args()

    # 性能优化建议
    torch.set_float32_matmul_precision('medium')

    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- 1. 加载配置 ---
    metadata_path = os.path.join(args.data_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # 默认训练配置
    training_config = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'lr_patience': 10,
        'batch_size': 8,
        'max_epochs': 100,
        'patience': 15,
        'grad_clip': 1.0,
        'use_fp16': True,
        'num_workers': 4
    }
    
    # 默认ESA模型配置
    esa_config = {
        'task_type': "regression",
        'graph_dim': 128,
        'linear_output_size': 1,
        'monitor_loss_name': "val_loss",
        'xformers_or_torch_attn': "xformers", # 启用 xformers
        'hidden_dims': [128, 128, 128],
        'num_heads': [8, 8, 8],
        'layer_types': ['M', 'S', 'M'], # 交错MAB和SAB
        'sab_dropout': 0.1,
        'mab_dropout': 0.1,
        'pma_dropout': 0.1,
        'apply_attention_on': "edge",
        'use_mlps': True,
        'mlp_type': "gated_mlp", # 启用 gated_mlp 以使用 flash-attn
        'norm_type': "LN",
        'regression_loss_fn': "mse",
        'posenc': ""
    }

    # --- 2. 准备数据加载器 ---
    train_dataset = Stage2Dataset(os.path.join(args.data_dir, 'train.pkl'))
    val_dataset = Stage2Dataset(os.path.join(args.data_dir, 'valid.pkl'))
    
    # 计算全局最大节点/边数
    all_node_counts = [data.num_nodes for data in train_dataset.data + val_dataset.data]
    all_edge_counts = [data.num_edges for data in train_dataset.data + val_dataset.data]
    
    # 关键修复：预先计算最终的填充尺寸，并将其传递给所有组件
    def nearest_multiple_of_8(n):
        return math.ceil(n / 8) * 8
        
    raw_max_nodes = max(all_node_counts) if all_node_counts else 0
    raw_max_edges = max(all_edge_counts) if all_edge_counts else 0
    
    # ESA模型内部会对 set_max_items + 1，所以我们在这里也这样做以保持一致
    final_max_nodes = nearest_multiple_of_8(raw_max_nodes + 1)
    final_max_edges = nearest_multiple_of_8(raw_max_edges + 1)
    
    esa_config['set_max_items'] = raw_max_edges # Estimator期望接收原始值
    
    collater = Collater(global_max_edges=final_max_edges, global_max_nodes=final_max_nodes)
    
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True, num_workers=training_config['num_workers'], collate_fn=collater)
    val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], shuffle=False, num_workers=training_config['num_workers'], collate_fn=collater)

    # --- 3. 初始化模型 ---
    model = PhysESA(
        esa_config=esa_config,
        training_config=training_config,
        graph_builder_config=metadata['graph_builder_config']
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
        precision=16 if training_config['use_fp16'] else 32,
        gradient_clip_val=training_config['grad_clip']
    )

    # --- 5. 开始训练 ---
    print("开始训练PhysESA模型...")
    trainer.fit(model, train_loader, val_loader)
    
    print(f"训练完成！模型保存在: {args.output_dir}")

if __name__ == "__main__":
    main()