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

from model.esa.masked_layers import Estimator
from molecular_graph import GraphBuilder

class MulScalePhysESA(pl.LightningModule):
    """
    A multi-scale wrapper that uses a two-stage ESA model.
    1. An atomic-level ESA operates on the fine-grained graph.
    2. A pooling layer aggregates atom features to group (residue/motif) features.
    3. A group-level ESA operates on the coarse-grained graph.
    """
    def __init__(self, atomic_esa_config: dict, coarse_esa_config: dict, training_config: dict, graph_builder_config: dict, pool_stop_layer: int = 3):
        """
        Args:
            atomic_esa_config (dict): Config for the atomic-level ESA model.
            coarse_esa_config (dict): Config for the coarse-grained ESA model.
            training_config (dict): Config for training, like learning rate.
            graph_builder_config (dict): Config for the GraphBuilder.
            pool_stop_layer (int): The atomic ESA layer index after which pooling occurs.
        """
        super().__init__()
        self.save_hyperparameters()

        self.training_config = training_config
        self.pool_stop_layer = pool_stop_layer

        # Instantiate the graph builder
        self.graph_builder = GraphBuilder(**graph_builder_config)
        feature_dims = self.graph_builder.get_feature_dimensions()

        # --- Atomic Level Model ---
        atomic_esa_config['num_features'] = feature_dims['node_dim']
        atomic_esa_config['edge_dim'] = feature_dims['edge_dim']
        self.atomic_model = Estimator(**atomic_esa_config)
        
        # We only use the encoder part of the atomic model up to a certain layer
        self.atomic_encoder = nn.Sequential(*self.atomic_model.encoder[:self.pool_stop_layer])

        # --- Coarse-Grained Level Model ---
        # The node features for the coarse model are the output of the atomic encoder's hidden dim
        # The edge features for the coarse model will be derived from group positions
        coarse_node_dim = atomic_esa_config['graph_dim']
        coarse_edge_dim = self.graph_builder.num_gaussians + 3 # Distance (GBF) + Direction
        
        coarse_esa_config['num_features'] = coarse_node_dim 
        coarse_esa_config['edge_dim'] = coarse_edge_dim
        self.coarse_model = Estimator(**coarse_esa_config)

        # --- Metrics ---
        self.setup_metrics()

    def setup_metrics(self):
        """Initializes validation and test metrics."""
        self.val_pearson = PearsonCorrCoef()
        self.val_spearman = SpearmanCorrCoef()
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_mae = MeanAbsoluteError()
        self.test_r2 = R2Score()
        self.test_rmse = MeanSquaredError(squared=False)
        self.test_mae = MeanAbsoluteError()
        self.test_pearson = PearsonCorrCoef()
        self.test_spearman = SpearmanCorrCoef()

    def _pool_to_coarse_graph(self, batch, atomic_edge_feats):
        """Pools atomic features to create the coarse-grained graph."""
        # 1. Derive updated atomic node features from edge features
        # A simple approach: for each node, average the features of incoming edges
        src, dst = batch.edge_index
        atomic_node_feats_from_edges = scatter_mean(atomic_edge_feats, dst, dim=0, dim_size=batch.num_nodes)
        
        # 2. Pool atomic node features to group node features
        group_node_feats = scatter_mean(atomic_node_feats_from_edges, batch.atom_to_group_idx, dim=0)
        
        # 3. Get group positions and build coarse edge features
        group_pos = scatter_mean(batch.pos, batch.atom_to_group_idx, dim=0)
        
        coarse_src, coarse_dst = batch.coarse_edge_index
        dist = torch.norm(group_pos[coarse_dst] - group_pos[coarse_src], dim=-1)
        
        dist_feats = self.graph_builder.gaussian_basis_functions(dist)
        dir_feats = (group_pos[coarse_dst] - group_pos[coarse_src]) / (dist.unsqueeze(-1) + 1e-8)
        
        coarse_edge_feats = torch.cat([dist_feats, dir_feats], dim=1)

        return group_node_feats, coarse_edge_feats

    def forward(self, batch: Batch) -> torch.Tensor:
        # --- 1. Atomic-Level Processing ---
        # Prepare initial edge representations for the atomic graph
        atomic_input = self.atomic_model.feature_pre_mlp(
            torch.cat([
                batch.x[batch.edge_index[0]], 
                batch.x[batch.edge_index[1]], 
                batch.edge_attr
            ], dim=-1)
        )
        # Run through the first few layers of the atomic encoder
        atomic_edge_feats = self.atomic_encoder(atomic_input)

        # --- 2. Pooling and Coarse Graph Construction ---
        group_node_feats, coarse_edge_feats = self._pool_to_coarse_graph(batch, atomic_edge_feats)

        # --- 3. Coarse-Grained Level Processing ---
        # Prepare input for the coarse model
        coarse_input = self.coarse_model.feature_pre_mlp(
            torch.cat([
                group_node_feats[batch.coarse_edge_index[0]],
                group_node_feats[batch.coarse_edge_index[1]],
                coarse_edge_feats
            ], dim=-1)
        )
        
        # Run through the full coarse-grained model
        # We need to pass the correct mask for the coarse graph
        coarse_batch_map = batch.atom_to_group_idx[batch.ptr[:-1]]
        predictions = self.coarse_model.forward_from_features(coarse_input, batch.coarse_edge_index, coarse_batch_map)
        
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
            'val/pearson': self.val_pearson.compute(),
            'val/spearman': self.val_spearman.compute(),
            'val/rmse': self.val_rmse.compute(),
            'val/mae': self.val_mae.compute()
        }, on_epoch=True, logger=True)
        self.val_pearson.reset()
        self.val_spearman.reset()
        self.val_rmse.reset()
        self.val_mae.reset()

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
        print("\n" + "="*30)
        print("      Multi-Scale Test Results      ")
        print("="*30)
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE:     {rmse:.4f}")
        print(f"  MAE:      {mae:.4f}")
        print(f"  Pearson:  {pearson:.4f}")
        print(f"  Spearman: {spearman:.4f}")
        print("="*30)
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
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }