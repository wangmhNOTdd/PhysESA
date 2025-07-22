"""
阶段一：训练脚本
简化版本的PhysESA模型训练
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import List, Dict, Tuple
import argparse
import json

# 添加model目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from molecular_graph import MolecularGraphBuilder
from simple_physesa import SimplePhysESA
from dataset import create_data_loaders

class Trainer:
    """简化的训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        log_dir: str = "./logs"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 优化器和损失函数
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # 日志
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.best_val_loss = float('inf')
        
        # 统计信息
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (graphs, targets) in enumerate(self.train_loader):
            # 注意：我们的collate_fn返回的是图列表，不是单个批次
            # 阶段一先处理单个图
            if len(graphs) > 0:
                graph = graphs[0]  # 取第一个图
                target = targets[0].to(self.device)
                
                # 将图数据移动到正确的设备
                graph = graph.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                
                try:
                    prediction = self.model(graph)
                    loss = self.criterion(prediction.squeeze(), target)
                    
                    # 反向传播
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if batch_idx % 50 == 0:
                        print(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                              f'Loss: {loss.item():.4f}, Pred: {prediction.item():.3f}, '
                              f'Target: {target.item():.3f}')
                        
                except Exception as e:
                    print(f"训练时出错 (batch {batch_idx}): {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def validate(self, epoch: int) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        predictions = []
        targets_list = []
        
        with torch.no_grad():
            for graphs, targets in self.val_loader:
                if len(graphs) > 0:
                    graph = graphs[0]
                    target = targets[0].to(self.device)
                    
                    # 将图数据移动到正确的设备
                    graph = graph.to(self.device)
                    
                    try:
                        prediction = self.model(graph)
                        loss = self.criterion(prediction.squeeze(), target)
                        
                        total_loss += loss.item()
                        num_batches += 1
                        
                        predictions.append(prediction.item())
                        targets_list.append(target.item())
                        
                    except Exception as e:
                        print(f"验证时出错: {e}")
                        continue
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # 计算相关系数
        if len(predictions) > 1:
            corr = np.corrcoef(predictions, targets_list)[0, 1]
            mae = np.mean(np.abs(np.array(predictions) - np.array(targets_list)))
            
            print(f'Validation - Epoch {epoch}: Loss: {avg_loss:.4f}, '
                  f'MAE: {mae:.4f}, Correlation: {corr:.4f}')
            
            # 记录到tensorboard
            self.writer.add_scalar('Val/Loss', avg_loss, epoch)
            self.writer.add_scalar('Val/MAE', mae, epoch)
            self.writer.add_scalar('Val/Correlation', corr, epoch)
        
        return avg_loss
    
    def train(self, num_epochs: int, save_dir: str = "./checkpoints"):
        """完整训练循环"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate(epoch)
            self.val_losses.append(val_loss)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 记录训练损失到tensorboard
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 记录日志信息到文件
            log_file = os.path.join(self.log_dir, 'training_log.txt')
            with open(log_file, 'a') as f:
                f.write(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, '
                       f'Val Loss: {val_loss:.4f}, '
                       f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}\n')
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
                print(f"保存最佳模型 (val_loss: {val_loss:.4f})")
            
            # 定期保存检查点
            if epoch % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'))
            
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch} 完成，用时: {epoch_time:.2f}s, '
                  f'训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}')
            print("-" * 60)
        
        # 保存训练历史
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        self.writer.close()
        print(f"训练完成！最佳验证损失: {self.best_val_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description="阶段一：PhysESA模型训练")
    parser.add_argument('--data_root', type=str, default='./datasets/pdbbind',
                        help='数据集根目录')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='最大样本数量（用于调试）')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='隐藏层维度')
    parser.add_argument('--num_mab_layers', type=int, default=4,
                        help='MAB层数量')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout率')
    parser.add_argument('--cutoff_radius', type=float, default=5.0,
                        help='原子相互作用截断半径')
    parser.add_argument('--num_gaussians', type=int, default=16,
                        help='高斯基函数数量')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练epoch数')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志目录')
    
    args = parser.parse_args()
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查数据目录
    if not os.path.exists(args.data_root):
        print(f"错误: 数据目录不存在: {args.data_root}")
        return
    
    try:
        # 创建数据加载器
        print("创建数据加载器...")
        train_loader, val_loader, test_loader = create_data_loaders(
            data_root=args.data_root,
            cutoff_radius=args.cutoff_radius,
            num_gaussians=args.num_gaussians,
            max_samples=args.max_samples,
            batch_size=1,  # 由于图大小差异很大，使用batch_size=1
            num_workers=0  # 避免多进程问题
        )
        
        print(f"数据加载完成:")
        print(f"  训练集: {len(train_loader.dataset)} 样本")
        print(f"  验证集: {len(val_loader.dataset)} 样本")
        print(f"  测试集: {len(test_loader.dataset)} 样本")
        
        # 获取特征维度
        builder = MolecularGraphBuilder(args.cutoff_radius, args.num_gaussians)
        feature_dims = builder.get_feature_dimensions()
        
        # 创建模型
        print("创建模型...")
        model = SimplePhysESA(
            node_dim=feature_dims['node_dim'],
            edge_dim=feature_dims['edge_dim'],
            hidden_dim=args.hidden_dim,
            num_mab_layers=args.num_mab_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            cutoff_radius=args.cutoff_radius
        )
        
        print(f"模型创建完成:")
        print(f"  节点特征维度: {feature_dims['node_dim']}")
        print(f"  边特征维度: {feature_dims['edge_dim']}")
        print(f"  隐藏层维度: {args.hidden_dim}")
        print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            log_dir=args.log_dir
        )
        
        # 开始训练
        trainer.train(
            num_epochs=args.num_epochs,
            save_dir=args.save_dir
        )
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
