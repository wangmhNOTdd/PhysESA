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
import math
from typing import List, Dict, Any
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
from torch_geometric.data import Data, Batch

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


def collate_fn(batch: List[Dict]) -> Batch:
    """
    自定义collate函数，为ESA模型准备标准格式的图数据
    直接使用预处理好的节点/边特征，而不是从edge_representations重构
    """
    data_list = []
    max_nodes = 0
    max_edges = 0

    for sample in batch:
        # 直接使用预处理好的特征
        node_features = sample['node_features']
        edge_features = sample['edge_features']
        edge_index = sample['edge_index']
        affinity = sample['affinity']
        num_nodes = sample['num_nodes']
        num_edges = sample['num_edges']

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            y=torch.tensor([affinity], dtype=torch.float32)
        )
        data_list.append(data)
        
        max_nodes = max(max_nodes, num_nodes)
        max_edges = max(max_edges, num_edges)

    # 使用torch_geometric的Batch类自动处理批处理
    batch_data = Batch.from_data_list(data_list)
    
    # 添加ESA需要的全局属性
    def nearest_multiple_of_8(n):
        return math.ceil(n / 8) * 8
    
    # 注意：这里的max_node/edge_global是整个批次中单个图的最大值，而不是总和
    batch_data.max_node_global = torch.tensor([nearest_multiple_of_8(max_nodes + 1)], dtype=torch.long)
    batch_data.max_edge_global = torch.tensor([nearest_multiple_of_8(max_edges + 1)], dtype=torch.long)
    
    return batch_data


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
        
        # 使用nearest_multiple_of_8确保与ESA模型内部计算一致
        def nearest_multiple_of_8(n):
            return math.ceil(n / 8) * 8
        
        raw_max_edges = max(all_edge_counts)
        self.max_edges = nearest_multiple_of_8(raw_max_edges + 1)  # +1 like in ESA model
        print(f"数据集最大边数: {raw_max_edges} -> 对齐后: {self.max_edges}")
        
    def create_model(self) -> Estimator:
        """创建ESA模型"""
        
        # 从元数据文件获取特征维度
        metadata_path = os.path.join(os.path.dirname(self.config['train_data_path']), 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        feature_dims = metadata['feature_dimensions']
        
        # 现在使用标准的节点-边图结构
        # 节点特征：42维（原子类型10维 + 位置编码32维）
        # 边特征：16维（Gaussian距离特征）
        node_dim = feature_dims['node_dim']  # 42
        edge_dim = feature_dims['edge_dim']  # 16
        
        print(f"模型配置:")
        print(f"  节点特征维度: {node_dim}")
        print(f"  边特征维度: {edge_dim}")
        print(f"  使用Edge-Set Attention")
        
        model = Estimator(
            task_type="regression",
            num_features=node_dim,  # 使用节点特征维度
            graph_dim=self.config['graph_dim'],
            edge_dim=edge_dim,  # 使用边特征维度
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
            norm_type="LN",  # 使用Layer Norm（ESA模型识别的格式）
            set_max_items=self.max_edges,
            early_stopping_patience=self.config['patience'],
            optimiser_weight_decay=self.config['weight_decay'],
            regression_loss_fn="mse",
            posenc="",  # 不使用ESA内置位置编码（我们在节点特征中已包含Fourier位置编码）
            posenc_dim=0  # ESA内置位置编码维度设为0
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
