"""
Physics-Informed 3D-ESA Model for Protein-Ligand Affinity Prediction
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.data import Batch
from torch.nn import functional as F
from torchmetrics import R2Score, MeanSquaredError, MeanAbsoluteError, PearsonCorrCoef, SpearmanCorrCoef

# Correctly import the Estimator which will act as our main model interface
from model.esa.masked_layers import Estimator
from molecular_graph import GraphBuilder

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
            graph_builder_config (dict): Configuration for the GraphBuilder.
        """
        super().__init__()
        self.save_hyperparameters()

        self.training_config = training_config
        
        # Instantiate the graph builder, which is not used in forward but might be needed for other hooks
        self.graph_builder = GraphBuilder(**graph_builder_config)
        
        # Get feature dimensions from the graph builder to configure the ESA model
        feature_dims = self.graph_builder.get_feature_dimensions()
        esa_config['num_features'] = feature_dims['node_dim']
        esa_config['edge_dim'] = feature_dims['edge_dim']

        # The core model logic is now fully encapsulated within the Estimator class.
        # PhysESA's role is to simply host it within the Lightning framework.
        self.esa_model = Estimator(**esa_config)

        # Metrics for validation
        self.val_pearson = PearsonCorrCoef()
        self.val_spearman = SpearmanCorrCoef()
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_mae = MeanAbsoluteError()

        # Metrics for testing
        self.test_r2 = R2Score()
        self.test_rmse = MeanSquaredError(squared=False)
        self.test_mae = MeanAbsoluteError()
        self.test_pearson = PearsonCorrCoef()
        self.test_spearman = SpearmanCorrCoef()

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