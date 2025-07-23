"""
阶段一训练脚本
使用ESA模型进行蛋白质-小分子亲和力预测
"""

import os
import pickle
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from typing import List, Dict, Any
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

# 添加model路径
sys.path.append('./model')
from esa.models import Estimator


class Stage1Dataset(Dataset):
    """阶段一数据集类"""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: pickle文件路径
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"加载数据集: {data_path}")
        print(f"样本数: {len(self.data)}")
        
        # 统计信息
        edge_counts = [sample['num_edges'] for sample in self.data]
        affinities = [sample['affinity'] for sample in self.data]
        
        print(f"边数统计: 均值={np.mean(edge_counts):.1f}, "
              f"范围=[{np.min(edge_counts)}, {np.max(edge_counts)}]")
        print(f"亲和力统计: 均值={np.mean(affinities):.2f}, "
              f"标准差={np.std(affinities):.2f}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch: List[Dict]) -> Dict:
    """
    自定义collate函数，处理变长的边表示
    """
    # 收集所有数据
    edge_representations = []
    affinities = []
    batch_indices = []
    edge_indices = []
    num_edges_list = []
    
    current_offset = 0
    
    for i, sample in enumerate(batch):
        edge_repr = sample['edge_representations']  # [num_edges, feature_dim]
        edge_representations.append(edge_repr)
        affinities.append(sample['affinity'])
        
        # 为每条边分配批次索引
        num_edges = edge_repr.shape[0]
        batch_indices.extend([i] * num_edges)
        num_edges_list.append(num_edges)
        
        # 更新边索引（加上偏移量）
        edge_index = sample['edge_index'] + current_offset
        edge_indices.append(edge_index)
        current_offset += sample['num_nodes']
    
    # 拼接所有边表示
    edge_representations = torch.cat(edge_representations, dim=0)  # [total_edges, feature_dim]
    batch_indices = torch.tensor(batch_indices, dtype=torch.long)   # [total_edges]
    affinities = torch.tensor(affinities, dtype=torch.float32)      # [batch_size]
    edge_indices = torch.cat(edge_indices, dim=1)                   # [2, total_edges]
    
    # 计算最大边数（用于padding）
    max_edges = max(num_edges_list)
    
    return {
        'x': edge_representations,           # [total_edges, feature_dim]
        'edge_index': edge_indices,          # [2, total_edges] 
        'batch': batch_indices,              # [total_edges]
        'y': affinities,                     # [batch_size]
        'num_max_items': max_edges,          # int
        'edge_attr': None                    # 在edge_representations中已包含
    }


class Stage1Trainer:
    """阶段一训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 加载数据集
        self.train_dataset = Stage1Dataset(config['train_data_path'])
        self.val_dataset = Stage1Dataset(config['val_data_path'])
        self.test_dataset = Stage1Dataset(config['test_data_path'])
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            collate_fn=collate_fn,
            persistent_workers=True if config['num_workers'] > 0 else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            collate_fn=collate_fn,
            persistent_workers=True if config['num_workers'] > 0 else False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            collate_fn=collate_fn,
            persistent_workers=True if config['num_workers'] > 0 else False
        )
        
        # 计算最大边数（用于设置模型参数）
        all_edge_counts = []
        for dataset in [self.train_dataset, self.val_dataset, self.test_dataset]:
            edge_counts = [sample['num_edges'] for sample in dataset.data]
            all_edge_counts.extend(edge_counts)
        
        self.max_edges = max(all_edge_counts)
        print(f"数据集最大边数: {self.max_edges}")
        
    def create_model(self) -> Estimator:
        """创建ESA模型"""
        
        # 从元数据文件获取特征维度
        metadata_path = os.path.join(os.path.dirname(self.config['train_data_path']), 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        feature_dims = metadata['feature_dimensions']
        
        # 计算输入特征维度：node_dim * 2 + edge_dim
        # (每条边由两个节点特征 + 边特征组成)
        input_dim = feature_dims['node_dim'] * 2 + feature_dims['edge_dim']
        
        print(f"模型配置:")
        print(f"  输入特征维度: {input_dim}")
        print(f"  节点特征维度: {feature_dims['node_dim']}")
        print(f"  边特征维度: {feature_dims['edge_dim']}")
        
        model = Estimator(
            task_type="regression",
            num_features=input_dim,
            graph_dim=self.config['graph_dim'],
            edge_dim=None,  # 边特征已经包含在输入中
            batch_size=self.config['batch_size'],
            lr=self.config['learning_rate'],
            linear_output_size=1,
            monitor_loss_name="val_loss",
            xformers_or_torch_attn="torch",  # 使用torch原生注意力
            hidden_dims=self.config['hidden_dims'],
            num_heads=[self.config['num_heads']] * len(self.config['layer_types']),  # 为每层指定头数
            num_sabs=self.config['num_sabs'],
            sab_dropout=self.config['sab_dropout'],
            mab_dropout=self.config['mab_dropout'],
            pma_dropout=self.config['pma_dropout'],
            apply_attention_on="edge",  # 使用ESA（边注意力）
            layer_types=self.config['layer_types'],  # 阶段一只使用MAB
            use_mlps=True,
            mlp_hidden_size=128,
            mlp_type="standard",
            norm_type="layer_norm",
            set_max_items=self.max_edges,
            early_stopping_patience=self.config['patience'],
            optimiser_weight_decay=self.config['weight_decay'],
            regression_loss_fn="mse",
            posenc=None,  # 不使用额外的位置编码（我们在输入中已包含）
            posenc_dim=0  # 位置编码维度设为0
        )
        
        return model
    
    def train(self):
        """训练模型"""
        
        # 创建模型
        model = self.create_model()
        
        # 设置回调
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config['output_dir'],
            filename='stage1-{epoch:02d}-{val_loss:.3f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=self.config['patience'],
            mode='min',
            verbose=True
        )
        
        # 设置logger
        logger = TensorBoardLogger(
            save_dir=self.config['output_dir'],
            name='stage1_logs'
        )
        
        # 创建trainer
        trainer = pl.Trainer(
            max_epochs=self.config['max_epochs'],
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=logger,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            precision=16 if self.config['use_fp16'] else 32,
            gradient_clip_val=self.config['grad_clip'],
            log_every_n_steps=10,
            check_val_every_n_epoch=1
        )
        
        print("开始训练...")
        print(f"设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        print(f"最大轮数: {self.config['max_epochs']}")
        print(f"批次大小: {self.config['batch_size']}")
        print(f"学习率: {self.config['learning_rate']}")
        
        # 训练
        trainer.fit(model, self.train_loader, self.val_loader)
        
        # 测试
        print("开始测试...")
        trainer.test(model, self.test_loader, ckpt_path='best')
        
        print(f"训练完成！模型保存在: {self.config['output_dir']}")
        
        return model, trainer


def create_default_config(data_dir: str, output_dir: str) -> Dict[str, Any]:
    """创建默认配置"""
    return {
        # 数据路径
        'train_data_path': os.path.join(data_dir, 'train.pkl'),
        'val_data_path': os.path.join(data_dir, 'valid.pkl'),
        'test_data_path': os.path.join(data_dir, 'test.pkl'),
        'output_dir': output_dir,
        
        # 训练参数
        'batch_size': 8,  # 较小的批次大小
        'learning_rate': 1e-4,
        'max_epochs': 100,
        'patience': 15,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'use_fp16': True,
        'num_workers': 4,
        
        # 模型参数（阶段一简化版）
        'graph_dim': 128,
        'hidden_dims': [128, 128],  # 2层，每层128维
        'num_heads': 8,
        'num_sabs': 0,  # 阶段一不使用SAB，只使用MAB
        'layer_types': ['M', 'M'],  # 只使用MAB层
        'sab_dropout': 0.1,
        'mab_dropout': 0.1,
        'pma_dropout': 0.1
    }


def main():
    parser = argparse.ArgumentParser(description='阶段一ESA模型训练')
    parser.add_argument('--data_dir', type=str, default='./experiments/stage1',
                       help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./experiments/stage1/checkpoints',
                       help='输出目录')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（可选）')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='最大训练轮数')
    parser.add_argument('--test_run', action='store_true',
                       help='测试运行（减少epochs）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查数据文件
    required_files = ['train.pkl', 'valid.pkl', 'test.pkl', 'metadata.json']
    for file in required_files:
        file_path = os.path.join(args.data_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到必需的文件: {file_path}")
    
    # 创建配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config(args.data_dir, args.output_dir)
    
    # 更新配置
    config.update({
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_epochs': args.max_epochs
    })
    
    # 测试运行模式
    if args.test_run:
        config['max_epochs'] = 5
        config['patience'] = 3
        print("*** 测试运行模式：仅训练5个epochs ***")
    
    # 保存配置
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("=== 阶段一训练配置 ===")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"批次大小: {config['batch_size']}")
    print(f"学习率: {config['learning_rate']}")
    print(f"最大轮数: {config['max_epochs']}")
    print(f"模型架构: ESA with {config['layer_types']}")
    
    # 创建训练器并开始训练
    trainer = Stage1Trainer(config)
    model, pl_trainer = trainer.train()
    
    print("阶段一训练完成！")


if __name__ == "__main__":
    main()
