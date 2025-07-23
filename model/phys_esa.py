"""
Physics-Informed 3D-ESA Model for Protein-Ligand Affinity Prediction
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.data import Data, Batch
from torch.nn import functional as F
import bitsandbytes as bnb
import torch_geometric

from esa.models import Estimator
from molecular_graph import Stage2GraphBuilder

class PhysESA(pl.LightningModule):
    """
    A wrapper model that integrates physics-informed feature engineering
    with the core ESA graph attention model.
    """
    def __init__(self, esa_config: dict, training_config: dict, graph_builder_config: dict):
        """
        Args:
            esa_config (dict): Configuration for the core ESA model (Estimator).
            training_config (dict): Configuration for training, like learning rate.
            graph_builder_config (dict): Configuration for the Stage2GraphBuilder.
        """
        super().__init__()
        self.save_hyperparameters()

        self.training_config = training_config
        
        # Instantiate the graph builder for on-the-fly feature engineering
        self.graph_builder = Stage2GraphBuilder(**graph_builder_config)
        
        # Get feature dimensions from the graph builder to configure the ESA model
        feature_dims = self.graph_builder.get_feature_dimensions()
        esa_config['num_features'] = feature_dims['node_dim']
        esa_config['edge_dim'] = feature_dims['edge_dim']

        # The core ESA model is instantiated here
        self.esa_model = Estimator(**esa_config)

        # Define a new output layer suitable for graph regression
        # Define a new output layer suitable for graph regression.
        # We fetch the dimensions from the instantiated esa_model to ensure correctness.
        input_dims = self.esa_model.num_inds * self.esa_model.hidden_dim
        self.output_layer = nn.Sequential(
            nn.Linear(input_dims, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        The forward pass for the PhysESA model.
        It directly uses the pre-computed graph features from the batch.
        """
        # 直接调用esa_model的内部forward，并传递正确的num_max_items
        # 我们不再依赖esa_model._step中的逻辑，以确保维度一致
        
        # 1. 准备ESA的输入
        x, edge_index, edge_attr, batch_mapping = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        num_max_items = batch.max_edge_global.item()

        # 2. 复刻Estimator.forward中的逻辑
        source = x[edge_index[0, :], :]
        target = x[edge_index[1, :], :]
        h = torch.cat((source, target), dim=1)

        if self.esa_model.edge_dim is not None and edge_attr is not None:
            h = torch.cat((h, edge_attr.float()), dim=1)

        h = self.esa_model.node_edge_mlp(h)

        edge_batch_index = batch_mapping.index_select(0, edge_index[0, :])
        h, _ = torch_geometric.utils.to_dense_batch(h, edge_batch_index, fill_value=0, max_num_nodes=num_max_items)
        
        # 3. 调用核心的ESA模块
        h = self.esa_model.st_fast(h, edge_index, batch_mapping, num_max_items=num_max_items)
        
        # 4. 得到最终预测
        # The output from ESA (st_fast) is (batch_size, num_inds, hidden_dim).
        # It needs to be flattened before being passed to the final MLP.
        h = torch.flatten(h, start_dim=1)
        
        # Use our custom output_layer instead of the one from Estimator
        predictions = self.output_layer(h)
        
        return predictions

    def _common_step(self, batch, batch_idx):
        """Common logic for training, validation, and test steps."""
        y_true = batch.y
        y_pred = self(batch)
        
        loss = F.mse_loss(y_pred.squeeze(), y_true.squeeze())
        return loss, y_pred, y_true

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.num_graphs)
        return {'val_loss': loss, 'y_pred': y_pred, 'y_true': y_true}

    def test_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        return {'test_loss': loss, 'y_pred': y_pred, 'y_true': y_true}

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = bnb.optim.AdamW8bit(
            self.parameters(),
            lr=self.training_config.get('learning_rate', 1e-4),
            weight_decay=self.training_config.get('weight_decay', 1e-5)
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=self.training_config.get('lr_patience', 10),
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }