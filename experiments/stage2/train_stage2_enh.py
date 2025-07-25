"""
阶段二增强版（多尺度）训练脚本
使用MulScalePhysESA模型进行蛋白质-小分子亲和力预测
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

from model.MulScalePhys_esa import MulScalePhysESA
from experiments.stage2.train_stage2 import Stage2Dataset, Collater, plot_training_results

def main():
    parser = argparse.ArgumentParser(description='阶段二多尺度PhysESA模型训练')
    parser.add_argument('--data_dir', type=str, default='./experiments/stage2', help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./experiments/stage2', help='输出和检查点目录')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径，用于测试模式')
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- 1. 加载配置 ---
    metadata_path = os.path.join(args.data_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    training_config = {
        'learning_rate': 1e-4, 'weight_decay': 1e-5, 'lr_patience': 10,
        'batch_size': 4, 'max_epochs': 150, 'patience': 20,
        'grad_clip': 1.0, 'use_fp16': False, 'num_workers': 8
    }
    
    # --- 2. 为多尺度模型定义两套ESA配置 ---
    # 原子层配置 (前3层)
    atomic_esa_config = {
        'graph_dim': 128, 'hidden_dims': [128, 128, 128],
        'num_heads': [8, 8, 8], 'layer_types': ['M', 'S', 'M'],
        'num_inds': 32, 'set_max_items': 0, 'linear_output_size': 1,
        'use_fp16': training_config['use_fp16'], 'node_or_edge': "edge",
        'xformers_or_torch_attn': "xformers", 'pre_or_post': "pre",
        'norm_type': "LN", 'sab_dropout': 0.1, 'mab_dropout': 0.1,
        'pma_dropout': 0.1, 'residual_dropout': 0.0, 'pma_residual_dropout': 0.0,
        'use_mlps': True, 'mlp_hidden_size': 128, 'num_mlp_layers': 2,
        'mlp_type': "gated_mlp", 'mlp_dropout': 0.1, 'use_mlp_ln': False,
    }

    # 基团/残基层配置 (后4层)
    coarse_esa_config = {
        'graph_dim': 128, 'hidden_dims': [128, 128, 128, 128],
        'num_heads': [8, 8, 8, 8], 'layer_types': ['M', 'S', 'P', 'S'],
        'num_inds': 32, 'set_max_items': 0, 'linear_output_size': 1,
        'use_fp16': training_config['use_fp16'], 'node_or_edge': "edge",
        'xformers_or_torch_attn': "xformers", 'pre_or_post': "pre",
        'norm_type': "LN", 'sab_dropout': 0.1, 'mab_dropout': 0.1,
        'pma_dropout': 0.1, 'residual_dropout': 0.0, 'pma_residual_dropout': 0.0,
        'use_mlps': True, 'mlp_hidden_size': 128, 'num_mlp_layers': 2,
        'mlp_type': "gated_mlp", 'mlp_dropout': 0.1, 'use_mlp_ln': False,
    }

    # --- 3. 准备数据加载器 ---
    train_dataset = Stage2Dataset(os.path.join(args.data_dir, 'train.pkl'))
    val_dataset = Stage2Dataset(os.path.join(args.data_dir, 'valid.pkl'))
    test_dataset = Stage2Dataset(os.path.join(args.data_dir, 'test.pkl'))
    
    all_edge_counts = [data.num_edges for data in train_dataset.data + val_dataset.data + test_dataset.data]
    raw_max_edges = max(all_edge_counts) if all_edge_counts else 0
    atomic_esa_config['set_max_items'] = raw_max_edges

    all_coarse_edge_counts = [data.coarse_edge_index.shape[1] for data in train_dataset.data + val_dataset.data + test_dataset.data]
    raw_max_coarse_edges = max(all_coarse_edge_counts) if all_coarse_edge_counts else 0
    coarse_esa_config['set_max_items'] = raw_max_coarse_edges

    # Collater不需要改变，因为它不处理特定于模型的填充
    collater = Collater(global_max_edges=0, global_max_nodes=0) # 动态填充在模型内部处理
    
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True, num_workers=training_config['num_workers'], collate_fn=collater)
    val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], shuffle=False, num_workers=training_config['num_workers'], collate_fn=collater)
    test_loader = DataLoader(test_dataset, batch_size=training_config['batch_size'], shuffle=False, num_workers=training_config['num_workers'], collate_fn=collater)

    # --- 4. 初始化模型 ---
    model = MulScalePhysESA(
        atomic_esa_config=atomic_esa_config,
        coarse_esa_config=coarse_esa_config,
        training_config=training_config,
        graph_builder_config=metadata['graph_builder_config']
    )

    # --- 5. 设置训练器 ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='mulscale-physesa-{epoch:02d}-{val_loss:.3f}',
        monitor='val_loss', mode='min', save_top_k=3
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss', patience=training_config['patience'], mode='min', verbose=True
    )
    logger = TensorBoardLogger(save_dir=args.output_dir, name='mulscale_physesa_logs')

    trainer = pl.Trainer(
        max_epochs=training_config['max_epochs'],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision="bf16" if training_config['use_fp16'] else 32,
        gradient_clip_val=training_config['grad_clip']
    )

    # --- 6. 开始训练或测试 ---
    if args.checkpoint:
        print(f"开始测试多尺度模型: {args.checkpoint}")
        trainer.test(model, dataloaders=test_loader, ckpt_path=args.checkpoint)
    else:
        print("开始训练多尺度PhysESA模型...")
        trainer.fit(model, train_loader, val_loader)
        print(f"训练完成！模型和日志保存在: {args.output_dir}")
        
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            print(f"\n使用最佳多尺度模型进行测试: {best_model_path}")
            trainer.test(model, dataloaders=test_loader, ckpt_path=best_model_path)
        else:
            print(f"\n警告: 未找到最佳模型检查点 '{best_model_path}'，跳过最终测试。")
    
    return args

if __name__ == "__main__":
    args = main()
    if args:
        log_dir = os.path.join(args.output_dir, 'mulscale_physesa_logs')
        if os.path.exists(log_dir):
            versions = sorted([d for d in os.listdir(log_dir) if d.startswith('version_')])
            if versions:
                latest_version_dir = os.path.join(log_dir, versions[-1])
                plot_training_results(latest_version_dir)