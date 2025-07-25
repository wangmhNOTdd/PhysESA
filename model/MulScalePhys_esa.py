"""
Multi-Scale Physics-Informed 3D-ESA Model for Protein-Ligand Affinity Prediction
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.data import Batch
from torch.nn import functional as F
from torch_scatter import scatter_mean
from torchmetrics import R2Score, MeanSquaredError, MeanAbsoluteError, PearsonCorrCoef, SpearmanCorrCoef

# Correctly import the Estimator wrapper class
from model.esa.masked_layers import Estimator
from molecular_graph import GraphBuilder

class MulScalePhysESA(pl.LightningModule):
    """
    A multi-scale wrapper that uses a two-stage Estimator model.
    1. An atomic-level Estimator processes the initial graph.
    2. Intermediate features are pooled to form a coarse-grained graph.
    3. A coarse-grained Estimator processes the new graph for the final prediction.
    """
    def __init__(self, atomic_esa_config: dict, coarse_esa_config: dict, training_config: dict, graph_builder_config: dict):
        super().__init__()
        self.save_hyperparameters()

        self.training_config = training_config
        
        # Graph builder is needed for coarse edge feature calculation
        self.graph_builder = GraphBuilder(**graph_builder_config)
        feature_dims = self.graph_builder.get_feature_dimensions()

        # --- Atomic Level Model ---
        # The Estimator handles its own feature preprocessing MLP
        atomic_model_config = atomic_esa_config.copy()
        atomic_model_config['num_features'] = feature_dims['node_dim']
        atomic_model_config['edge_dim'] = feature_dims['edge_dim']
        self.atomic_model = Estimator(**atomic_model_config)

        # --- Coarse-Grained Level Model ---
        # The input features for the coarse model are the output of the atomic encoder
        coarse_model_config = coarse_esa_config.copy()
        # Node features are the output of the atomic encoder's last layer
        coarse_node_dim = atomic_esa_config['hidden_dims'][-1] 
        # Edge features are calculated from distances and directions
        coarse_edge_dim = self.graph_builder.num_gaussians + 3 
        coarse_model_config['num_features'] = coarse_node_dim
        coarse_model_config['edge_dim'] = coarse_edge_dim
        self.coarse_model = Estimator(**coarse_model_config)

        self.setup_metrics()

    def setup_metrics(self):
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
        # --- 1. Atomic-Level Processing ---
        # Get intermediate features from the atomic model's encoder by using the flag
        atomic_edge_feats = self.atomic_model(batch, return_edge_features_before_pma=True)

        # --- 2. Pooling and Coarse Graph Construction ---
        # Pool edge features to nodes, then nodes to groups.
        # The features of an edge are pooled to its target node.
        atomic_node_feats = scatter_mean(atomic_edge_feats, batch.edge_index[1], dim=0, dim_size=batch.num_nodes)
        group_node_feats = scatter_mean(atomic_node_feats, batch.atom_to_group_idx, dim=0)
        
        # Also pool atom positions to get group center-of-mass
        group_pos = scatter_mean(batch.pos, batch.atom_to_group_idx, dim=0)
        
        # --- 3. Construct Coarse Graph Batch ---
        # Calculate coarse edge features (distance + direction)
        coarse_src_pos = group_pos[batch.coarse_edge_index[0]]
        coarse_dst_pos = group_pos[batch.coarse_edge_index[1]]
        dist = torch.norm(coarse_dst_pos - coarse_src_pos, p=2, dim=-1).unsqueeze(-1)
        dist_feats = self.graph_builder.gaussian_basis_functions(dist)
        dir_feats = F.normalize(coarse_dst_pos - coarse_src_pos, p=2, dim=-1)
        coarse_edge_attr = torch.cat([dist_feats, dir_feats], dim=-1)

        # Create a batch mapping for the coarse graph.
        # The batch index for each group is the same as the batch index of its constituent atoms.
        group_batch_mapping = scatter_mean(batch.batch.float(), batch.atom_to_group_idx, dim=0).long()

        coarse_batch = Batch(
            x=group_node_feats,
            edge_index=batch.coarse_edge_index,
            edge_attr=coarse_edge_attr,
            batch=group_batch_mapping,
            pos=group_pos,
            num_graphs=batch.num_graphs
        )

        # --- 4. Coarse-Grained Level Processing ---
        predictions = self.coarse_model(coarse_batch)
        
        return predictions

    def _common_step(self, batch, batch_idx):
        y_true = batch.y
        y_pred = self(batch)
        loss = F.mse_loss(y_pred, y_true.squeeze())
        return loss, y_pred, y_true

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.num_graphs)
        self.val_pearson.update(y_pred, y_true.squeeze())
        self.val_spearman.update(y_pred, y_true.squeeze())
        self.val_rmse.update(y_pred, y_true.squeeze())
        self.val_mae.update(y_pred, y_true.squeeze())
        return loss

    def on_validation_epoch_end(self):
        self.log_dict({
            'val/pearson': self.val_pearson.compute(), 'val/spearman': self.val_spearman.compute(),
            'val/rmse': self.val_rmse.compute(), 'val/mae': self.val_mae.compute()
        }, on_epoch=True, logger=True)
        self.val_pearson.reset(); self.val_spearman.reset(); self.val_rmse.reset(); self.val_mae.reset()

    def test_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        self.test_r2.update(y_pred, y_true.squeeze())
        self.test_rmse.update(y_pred, y_true.squeeze())
        self.test_mae.update(y_pred, y_true.squeeze())
        self.test_pearson.update(y_pred, y_true.squeeze())
        self.test_spearman.update(y_pred, y_true.squeeze())
        return loss

    def on_test_epoch_end(self):
        r2 = self.test_r2.compute()
        rmse = self.test_rmse.compute()
        mae = self.test_mae.compute()
        pearson = self.test_pearson.compute()
        spearman = self.test_spearman.compute()
        self.log_dict({
            'test/r2': r2, 'test/rmse': rmse, 'test/mae': mae,
            'test/pearson': pearson, 'test/spearman': spearman
        }, logger=True)
        print("\n" + "="*30); print("      Multi-Scale Test Results      "); print("="*30)
        print(f"  R² Score: {r2:.4f}"); print(f"  RMSE:     {rmse:.4f}"); print(f"  MAE:      {mae:.4f}")
        print(f"  Pearson:  {pearson:.4f}"); print(f"  Spearman: {spearman:.4f}"); print("="*30)
        self.test_r2.reset(); self.test_rmse.reset(); self.test_mae.reset()
        self.test_pearson.reset(); self.test_spearman.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.training_config['learning_rate'], weight_decay=self.training_config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=self.training_config.get('lr_patience', 5)
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}