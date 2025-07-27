"""
Physics-Informed 3D-ESA Model for Protein-Ligand Affinity Prediction
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.data import Batch, Data
from torch.nn import functional as F
from torchmetrics import R2Score, MeanSquaredError, MeanAbsoluteError, PearsonCorrCoef, SpearmanCorrCoef
from torch_scatter import scatter_sum
import copy

from model.esa.masked_layers import Estimator
from molecular_graph import MultiScaleGraphBuilder # Updated import

class PhysESA(pl.LightningModule):
    """
    多尺度PhysESA模型。
    该模型包含两个主要部分：
    1. 一个在原子图上操作的编码器 (atomic_encoder)。
    2. 一个在粗粒度图（残基/官能团）上操作的编码器+池化层 (coarse_encoder)。
    """
    def __init__(self, esa_config: dict, training_config: dict, feature_dims: dict):
        """
        Args:
            esa_config (dict): 包含两个尺度模型配置的字典。
            training_config (dict): 训练配置。
            feature_dims (dict): 从metadata.json加载的特征维度。
        """
        super().__init__()
        self.save_hyperparameters()

        self.training_config = training_config
        
        # --- 原子级别编码器配置 ---
        atomic_config = copy.deepcopy(esa_config)
        atomic_config['layer_types'] = ['M', 'M', 'S']
        atomic_config['hidden_dims'] = [128, 128, 128]
        atomic_config['num_heads'] = [8, 8, 8]
        # 原子编码器没有PMA池化层，因此num_inds和linear_output_size无效
        atomic_config.pop('num_inds', None)
        atomic_config.pop('linear_output_size', None)
        atomic_config['num_features'] = feature_dims['node_dim']
        atomic_config['edge_dim'] = feature_dims['edge_dim']
        atomic_config['set_max_items'] = esa_config['atomic_set_max_items']
        # 清理临时的键
        atomic_config.pop('atomic_set_max_items', None)
        atomic_config.pop('coarse_set_max_items', None)
        
        self.atomic_encoder = Estimator(**atomic_config)

        # --- 粗粒度级别编码器配置 ---
        coarse_config = copy.deepcopy(esa_config)
        coarse_config['layer_types'] = ['M', 'S', 'P', 'S']
        coarse_config['hidden_dims'] = [128, 128, 128, 128]
        coarse_config['num_heads'] = [8, 8, 8, 8]
        coarse_config['num_features'] = atomic_config['graph_dim']
        coarse_config['edge_dim'] = 0
        coarse_config['set_max_items'] = esa_config['coarse_set_max_items']
        # 清理临时的键
        coarse_config.pop('atomic_set_max_items', None)
        coarse_config.pop('coarse_set_max_items', None)
        
        self.coarse_encoder = Estimator(**coarse_config)

        # Metrics
        self.val_pearson = PearsonCorrCoef()
        self.val_spearman = SpearmanCorrCoef()
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_mae = MeanAbsoluteError()
        self.test_r2 = R2Score()
        self.test_rmse = MeanSquaredError(squared=False)
        self.test_mae = MeanAbsoluteError()
        self.test_pearson = PearsonCorrCoef()
        self.test_spearman = SpearmanCorrCoef()

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        执行多尺度前向传播。
        """
        # --- 1. 原子级别编码 ---
        # Estimator的forward返回的是处理后的边表示（在被to_dense_batch之前）
        # 我们需要修改Estimator来返回这个中间结果
        atomic_edge_embeds, atomic_edge_batch_map = self.atomic_encoder(batch, return_embeds=True)

        # --- 2. 从原子到粗粒度的池化 ---
        # 使用scatter_sum将原子级别的边表示聚合到粗粒度节点上
        # 注意：这里我们是把“边”的表示聚合为“节点”的表示
        coarse_node_features = scatter_sum(atomic_edge_embeds, atomic_edge_batch_map, dim=0)
        
        # 确保池化后的节点数与粗粒度图的节点数一致
        num_expected_coarse_nodes = batch.coarse_pos.shape[0]
        if coarse_node_features.shape[0] < num_expected_coarse_nodes:
            pad_size = num_expected_coarse_nodes - coarse_node_features.shape[0]
            padding = torch.zeros(pad_size, coarse_node_features.shape[1], device=self.device)
            coarse_node_features = torch.cat([coarse_node_features, padding], dim=0)

        # --- 3. 构建粗粒度图的Batch对象 ---
        coarse_batch = Batch(
            x=coarse_node_features,
            edge_index=batch.coarse_edge_index,
            edge_attr=None, # 粗粒度图没有显式边特征
            batch=batch.coarse_batch, # 需要在数据准备阶段创建这个映射
            pos=batch.coarse_pos,
            max_edge_global=batch.coarse_max_edge_global, # 需要在数据准备阶段计算
            max_node_global=batch.coarse_max_node_global  # 需要在数据准备阶段计算
        )

        # --- 4. 粗粒度级别编码和预测 ---
        predictions = self.coarse_encoder(coarse_batch)
        
        return predictions

    def _common_step(self, batch, batch_idx):
        """Common logic for training, validation, and test steps."""
        y_true = batch.y
        # The output of forward is already squeezed, so we use it directly.
        y_pred = self(batch)
        
        loss = F.mse_loss(y_pred, y_true.squeeze())
        return loss, y_pred, y_true

    def training_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.num_graphs)
        
        # Update validation metrics
        self.val_pearson.update(y_pred, y_true.squeeze())
        self.val_spearman.update(y_pred, y_true.squeeze())
        self.val_rmse.update(y_pred, y_true.squeeze())
        self.val_mae.update(y_pred, y_true.squeeze())
        
        return loss

    def on_validation_epoch_end(self):
        """
        Compute and log final validation metrics at the end of the validation epoch.
        """
        self.log_dict({
            'val/pearson': self.val_pearson.compute(),
            'val/spearman': self.val_spearman.compute(),
            'val/rmse': self.val_rmse.compute(),
            'val/mae': self.val_mae.compute()
        }, on_epoch=True, logger=True)
        
        # Reset metrics
        self.val_pearson.reset()
        self.val_spearman.reset()
        self.val_rmse.reset()
        self.val_mae.reset()

    def test_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        
        # Update metrics
        self.test_r2.update(y_pred, y_true.squeeze())
        self.test_rmse.update(y_pred, y_true.squeeze())
        self.test_mae.update(y_pred, y_true.squeeze())
        self.test_pearson.update(y_pred, y_true.squeeze())
        self.test_spearman.update(y_pred, y_true.squeeze())
        
        return loss

    def on_test_epoch_end(self):
        """
        Compute and log final metrics at the end of the test epoch.
        """
        r2 = self.test_r2.compute()
        rmse = self.test_rmse.compute()
        mae = self.test_mae.compute()
        pearson = self.test_pearson.compute()
        spearman = self.test_spearman.compute()

        self.log_dict({
            'test/r2': r2,
            'test/rmse': rmse,
            'test/mae': mae,
            'test/pearson': pearson,
            'test/spearman': spearman
        }, logger=True)

        print("\n" + "="*30)
        print("      Test Results      ")
        print("="*30)
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE:     {rmse:.4f}")
        print(f"  MAE:      {mae:.4f}")
        print(f"  Pearson:  {pearson:.4f}")
        print(f"  Spearman: {spearman:.4f}")
        print("="*30)

        # Reset metrics
        self.test_r2.reset()
        self.test_rmse.reset()
        self.test_mae.reset()
        self.test_pearson.reset()
        self.test_spearman.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }