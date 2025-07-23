"""
Physics-Informed 3D-ESA Model for Protein-Ligand Affinity Prediction
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.data import Batch
from torch.nn import functional as F

# Correctly import the Estimator which will act as our main model interface
from model.esa.masked_layers import Estimator
from molecular_graph import Stage2GraphBuilder

class PhysESA(pl.LightningModule):
    """
    A streamlined wrapper that integrates the graph builder with the core ESA model.
    This class is responsible for the PyTorch Lightning training loop and logic.
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
        
        # Instantiate the graph builder, which is not used in forward but might be needed for other hooks
        self.graph_builder = Stage2GraphBuilder(**graph_builder_config)
        
        # Get feature dimensions from the graph builder to configure the ESA model
        feature_dims = self.graph_builder.get_feature_dimensions()
        esa_config['num_features'] = feature_dims['node_dim']
        esa_config['edge_dim'] = feature_dims['edge_dim']

        # The core model logic is now fully encapsulated within the Estimator class.
        # PhysESA's role is to simply host it within the Lightning framework.
        self.esa_model = Estimator(**esa_config)

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        The forward pass is now greatly simplified.
        We delegate the entire graph processing and prediction logic to the esa_model.
        """
        # The Estimator's forward pass will handle everything.
        predictions = self.esa_model(batch)
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
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        return loss

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
            patience=5,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }