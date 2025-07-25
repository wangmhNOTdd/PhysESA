"""
Multi-Scale Physics-Informed 3D-ESA Model for Protein-Ligand Affinity Prediction
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from torch.nn import functional as F
from torch_scatter import scatter_mean
from torchmetrics import R2Score, MeanSquaredError, MeanAbsoluteError, PearsonCorrCoef, SpearmanCorrCoef

from model.esa.masked_layers import ESA
from model.esa.mlp_utils import SmallMLP
from molecular_graph import GraphBuilder

class MulScalePhysESA(pl.LightningModule):
    """
    A multi-scale wrapper that uses a two-stage ESA model.
    This implementation directly uses the core ESA module, bypassing the Estimator wrapper.
    """
    def __init__(self, atomic_esa_config: dict, coarse_esa_config: dict, training_config: dict, graph_builder_config: dict, pool_stop_layer: int = 3):
        super().__init__()
        self.save_hyperparameters()

        self.training_config = training_config
        self.pool_stop_layer = pool_stop_layer

        self.graph_builder = GraphBuilder(**graph_builder_config)
        feature_dims = self.graph_builder.get_feature_dimensions()

        # --- Atomic Level Model ---
        self.atomic_feature_pre_mlp = SmallMLP(
            in_dim=feature_dims['node_dim'] * 2 + feature_dims['edge_dim'],
            out_dim=atomic_esa_config['hidden_dims'][0],
            inter_dim=atomic_esa_config['hidden_dims'][0] * 2,
            num_layers=2, dropout_p=0.1
        )
        # We only need the encoder part of the atomic ESA
        atomic_encoder_config = atomic_esa_config.copy()
        atomic_encoder_config['layer_types'] = atomic_encoder_config['layer_types'][:pool_stop_layer]
        atomic_encoder_config['hidden_dims'] = atomic_encoder_config['hidden_dims'][:pool_stop_layer]
        atomic_encoder_config['num_heads'] = atomic_encoder_config['num_heads'][:pool_stop_layer]
        atomic_encoder_config.pop('graph_dim', None) # Remove key if it exists
        # 修复: ESA 期望 'dim_hidden', 而不是 'hidden_dims'
        atomic_encoder_config['dim_hidden'] = atomic_encoder_config.pop('hidden_dims')
        self.atomic_esa_encoder = ESA(
            num_outputs=1, dim_output=1, **atomic_encoder_config
        )

        # --- Coarse-Grained Level Model ---
        coarse_node_dim = atomic_esa_config['hidden_dims'][pool_stop_layer - 1]
        coarse_edge_dim = self.graph_builder.num_gaussians + 3
        self.coarse_feature_pre_mlp = SmallMLP(
            in_dim=coarse_node_dim * 2 + coarse_edge_dim,
            out_dim=coarse_esa_config['hidden_dims'][0],
            inter_dim=coarse_esa_config['hidden_dims'][0] * 2,
            num_layers=2, dropout_p=0.1
        )
        coarse_esa_config.pop('graph_dim', None) # Remove key if it exists
        # 修复: ESA 期望 'dim_hidden', 而不是 'hidden_dims'
        coarse_esa_config['dim_hidden'] = coarse_esa_config.pop('hidden_dims')
        self.coarse_esa = ESA(**coarse_esa_config)

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
        source = batch.x[batch.edge_index[0, :], :]
        target = batch.x[batch.edge_index[1, :], :]
        h_atomic = torch.cat([source, target, batch.edge_attr.float()], dim=1)
        h_atomic = self.atomic_feature_pre_mlp(h_atomic)
        
        edge_batch_index = batch.batch.index_select(0, batch.edge_index[0, :])
        h_atomic_dense, h_atomic_mask = to_dense_batch(h_atomic, edge_batch_index, fill_value=0)
        
        # Pass through atomic encoder
        atomic_edge_feats_dense = self.atomic_esa_encoder(
            h_atomic_dense, batch.edge_index, batch.batch, num_max_items=h_atomic_dense.shape[1],
            return_edge_features_before_pma=True
        )
        atomic_edge_feats = atomic_edge_feats_dense[h_atomic_mask]

        # --- 2. Pooling and Coarse Graph Construction ---
        atomic_node_feats = scatter_mean(atomic_edge_feats, batch.edge_index[1], dim=0, dim_size=batch.num_nodes)
        group_node_feats = scatter_mean(atomic_node_feats, batch.atom_to_group_idx, dim=0)
        group_pos = scatter_mean(batch.pos, batch.atom_to_group_idx, dim=0)
        
        coarse_src_pos, coarse_dst_pos = group_pos[batch.coarse_edge_index[0]], group_pos[batch.coarse_edge_index[1]]
        dist = torch.norm(coarse_dst_pos - coarse_src_pos, dim=-1)
        dist_feats = self.graph_builder.gaussian_basis_functions(dist)
        dir_feats = F.normalize(coarse_dst_pos - coarse_src_pos, p=2, dim=-1)
        coarse_edge_feats = torch.cat([dist_feats, dir_feats], dim=1)

        # --- 3. Coarse-Grained Level Processing ---
        coarse_src_feats = group_node_feats[batch.coarse_edge_index[0]]
        coarse_dst_feats = group_node_feats[batch.coarse_edge_index[1]]
        h_coarse = torch.cat([coarse_src_feats, coarse_dst_feats, coarse_edge_feats], dim=1)
        h_coarse = self.coarse_feature_pre_mlp(h_coarse)

        # Need to create a batch mapping for the coarse graph
        num_groups = group_node_feats.shape[0]
        _, group_batch_mapping = torch.unique(batch.atom_to_group_idx, return_inverse=True)
        group_batch_mapping = scatter_mean(batch.batch.float(), batch.atom_to_group_idx, dim=0).long()

        h_coarse_dense, _ = to_dense_batch(h_coarse, group_batch_mapping.index_select(0, batch.coarse_edge_index[0]))

        predictions = self.coarse_esa(
            h_coarse_dense, batch.coarse_edge_index, group_batch_mapping, num_max_items=h_coarse_dense.shape[1]
        )
        
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
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}