"""
Physics-Informed 3D-ESA Model for Protein-Ligand Affinity Prediction
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.data import Data, Batch
from torch.nn import functional as F
import bitsandbytes as bnb

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

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        The forward pass for the PhysESA model.
        It directly uses the pre-computed graph features from the batch.
        """
        predictions = self.esa_model(
            x=batch.x,
            edge_index=batch.edge_index,
            batch_mapping=batch.batch,
            edge_attr=batch.edge_attr,
            num_max_items=batch.max_edge_global.item(),
            batch=batch
        )
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