import torch
import torch_geometric
import math
import numpy as np
import pytorch_lightning as pl
import bitsandbytes as bnb

from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch
from collections import defaultdict
from typing import Optional, List

from utils.norm_layers import BN, LN
from esa.masked_layers import ESA
from esa.mlp_utils import SmallMLP, GatedMLPMulti

from data_loading.gaussian import GaussianLayer

from utils.reporting import (
    get_cls_metrics_binary_pt,
    get_cls_metrics_multilabel_pt,
    get_cls_metrics_multiclass_pt,
    get_regr_metrics_pt,
)

from utils.posenc_encoders.laplace_pos_encoder import LapPENodeEncoder
from utils.posenc_encoders.kernel_pos_encoder import KernelPENodeEncoder

# Task names for the LRGB peptides-func benchmark
pept_struct_target_names = ["Inertia_mass_a", "Inertia_mass_b", "Inertia_mass_c",
                        "Inertia_valence_a", "Inertia_valence_b",
                        "Inertia_valence_c", "length_a", "length_b", "length_c",
                        "Spherocity", "Plane_best_fit"]


def nearest_multiple_of_8(n):
    return math.ceil(n / 8) * 8


class Estimator(pl.LightningModule):
    def __init__(
        self,
        task_type: str,
        num_features: int,
        graph_dim: int,
        edge_dim: int,
        batch_size: int = 32,
        lr: float = 0.001,
        linear_output_size: int = 1,
        scaler=None,
        monitor_loss_name: str = "val_loss",
        xformers_or_torch_attn: str = "xformers",
        hidden_dims: List[int] = None,
        num_heads: int = None,
        num_sabs: int = None,
        sab_dropout: float = 0.0,
        mab_dropout: float = 0.0,
        pma_dropout: float = 0.0,
        apply_attention_on: str = "edge",
        layer_types: List[str] = None,
        use_mlps: bool = False,
        set_max_items: int=0,
        regression_loss_fn: str = "mae",
        early_stopping_patience: int = 30,
        optimiser_weight_decay: float = 1e-3,
        mlp_hidden_size: int = 64,
        mlp_type: str = "standard",
        attn_residual_dropout: float = 0.0,
        norm_type: str = "LN",
        triu_attn_mask: bool = False,
        output_save_dir: str = None,
        use_bfloat16: bool = True,
        is_node_task: bool = False,
        train_mask = None,
        val_mask = None,
        test_mask = None,
        posenc: str = None,
        num_mlp_layers: int = 3,
        pre_or_post: str = "pre",
        pma_residual_dropout: float = 0,
        use_mlp_ln: bool = False,
        mlp_dropout: float = 0,
        **kwargs,
    ):
        super().__init__()
        assert task_type in ["binary_classification", "multi_classification", "regression"]

        self.graph_dim = graph_dim
        self.task_type = task_type
        self.edge_dim = edge_dim

        self.num_features = num_features
        self.lr = lr
        self.batch_size = batch_size
        self.scaler = scaler
        self.linear_output_size = linear_output_size
        self.monitor_loss_name = monitor_loss_name
        self.mlp_hidden_size = mlp_hidden_size
        self.norm_type = norm_type
        self.set_max_items = set_max_items
        self.output_save_dir = output_save_dir
        self.is_node_task = is_node_task
        self.use_mlp_ln = use_mlp_ln
        self.pre_or_post = pre_or_post
        self.mlp_dropout = mlp_dropout

        self.use_mlps = use_mlps
        self.early_stopping_patience = early_stopping_patience
        self.optimiser_weight_decay = optimiser_weight_decay
        self.regression_loss_fn = regression_loss_fn
        self.mlp_type = mlp_type
        self.attn_residual_dropout = attn_residual_dropout
        self.pma_residual_dropout = pma_residual_dropout
        self.triu_attn_mask = triu_attn_mask
        self.use_bfloat16 = use_bfloat16
        self.layer_types = layer_types

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        # Store model outputs per epoch (for train, valid) or test run; used to compute the reporting metrics
        self.train_output = defaultdict(list)
        self.val_output = defaultdict(list)
        self.val_test_output = defaultdict(list)
        self.test_output = defaultdict(list)

        self.val_preds = defaultdict(list)

        self.test_true = defaultdict(list)
        self.val_true = defaultdict(list)

        # Keep track of how many times we called test
        self.num_called_test = 1

        # Metrics per epoch (for train, valid); for test use above variable to register metrics per test-run
        self.train_metrics = {}
        self.val_metrics = {}
        self.val_test_metrics = {}
        self.test_metrics = {}

        # Holds final graphs embeddings
        self.test_graph_embeddings = defaultdict(list)
        self.val_graph_embeddings = defaultdict(list)
        self.train_graph_embeddings = defaultdict(list)

        self.xformers_or_torch_attn = xformers_or_torch_attn
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.num_sabs = num_sabs
        self.sab_dropout = sab_dropout
        self.mab_dropout = mab_dropout
        self.pma_dropout = pma_dropout
        self.apply_attention_on = apply_attention_on
        self.posenc = posenc
        self.num_mlp_layers = num_mlp_layers

        self.rwse_encoder = None
        self.lap_encoder = None

        if "RWSE" in self.posenc:
            self.rwse_encoder = KernelPENodeEncoder()
        if "LapPE" in self.posenc:
            self.lap_encoder = LapPENodeEncoder()
 
        if self.norm_type == "BN":
            norm_fn = BN
        elif self.norm_type == "LN":
            norm_fn = LN

        if self.apply_attention_on == "node":
            in_dim = self.num_features
            if self.rwse_encoder is not None:
                in_dim += 24
            if self. lap_encoder is not None:
                in_dim += 4
            
            if self.mlp_type in ["standard", "gated_mlp"]:
                self.node_mlp = SmallMLP(
                    in_dim=in_dim,
                    inter_dim=128,
                    out_dim=self.hidden_dims[0],
                    use_ln=False,
                    dropout_p=0,
                    num_layers=self.num_mlp_layers if self.num_mlp_layers > 1 else self.num_mlp_layers + 1,
                )

            # Uncomment if you want the gated MLP here
            # elif self.mlp_type == "gated_mlp":
            #     self.node_mlp = GatedMLPMulti(
            #         in_dim=in_dim,
            #         out_dim=self.hidden_dims[0],
            #         inter_dim=128,
            #         activation=F.silu,
            #         dropout_p=0,
            #         num_layers=self.num_mlp_layers,
            #     )


        elif self.apply_attention_on == "edge":
            in_dim = self.num_features
            if self.rwse_encoder is not None:
                in_dim += 24
            if self.lap_encoder is not None:
                in_dim += 4

            in_dim = in_dim * 2
            if self.edge_dim is not None:
                in_dim += self.edge_dim
            
            if self.mlp_type in ["standard", "gated_mlp"]:
                self.node_edge_mlp = SmallMLP(
                    in_dim=in_dim,
                    inter_dim=128,
                    out_dim=self.hidden_dims[0],
                    use_ln=False,
                    dropout_p=0,
                    num_layers=self.num_mlp_layers,
                )

            # Uncomment if you want the gated MLP here

            # elif self.mlp_type == "gated_mlp":
            #     self.node_edge_mlp = GatedMLPMulti(
            #         in_dim=in_dim,
            #         out_dim=self.hidden_dims[0],
            #         inter_dim=128,
            #         activation=F.silu,
            #         dropout_p=0,
            #         num_layers=self.num_mlp_layers,
            #     )
            

        self.mlp_norm = norm_fn(self.hidden_dims[0])

        st_args = dict(
            num_outputs=32, # this is the k for PMA
            dim_output=self.graph_dim,
            xformers_or_torch_attn=self.xformers_or_torch_attn,
            dim_hidden=self.hidden_dims,
            num_heads=self.num_heads,
            sab_dropout=self.sab_dropout,
            mab_dropout=self.mab_dropout,
            pma_dropout=self.pma_dropout,
            use_mlps=self.use_mlps,
            mlp_hidden_size=self.mlp_hidden_size,
            mlp_type=self.mlp_type,
            norm_type=self.norm_type,
            node_or_edge=self.apply_attention_on,
            residual_dropout=self.attn_residual_dropout,
            set_max_items=nearest_multiple_of_8(self.set_max_items + 1),
            use_bfloat16=self.use_bfloat16,
            layer_types=self.layer_types,
            num_mlp_layers=self.num_mlp_layers,
            pre_or_post=self.pre_or_post,
            pma_residual_dropout=self.pma_residual_dropout,
            use_mlp_ln=self.use_mlp_ln,
            mlp_dropout=self.mlp_dropout,
        )

        self.st_fast = ESA(**st_args)

        if self.mlp_type in ["standard", "gated_mlp"]:
            self.output_mlp = SmallMLP(
                in_dim=self.graph_dim,
                inter_dim=128,
                out_dim=self.linear_output_size,
                use_ln=False,
                dropout_p=0,
                num_layers=self.num_mlp_layers if self.num_mlp_layers > 1 else self.num_mlp_layers + 1,
            )

            self.output_mlp = nn.Linear(self.graph_dim, self.linear_output_size)

        # Uncomment if you want the gated MLP here
            
        # elif self.mlp_type == "gated_mlp":
        #     self.output_mlp = GatedMLPMulti(
        #         in_dim=self.graph_dim,
        #         out_dim=self.linear_output_size,
        #         inter_dim=128,
        #         activation=F.silu,
        #         dropout_p=0,
        #         num_layers=self.num_mlp_layers,
        #     )
            
        self.atom_types = 64
        self.edge_types = 64 * 64

        self.atom_encoder = nn.Embedding(self.atom_types, self.embed_dim, padding_idx=0)
        self.tag_encoder = nn.Embedding(3, self.embed_dim)
        self.edge_proj = nn.Linear(self.k, self.embed_dim)

        self.gbf = GaussianLayer(self.k, self.edge_types)

        self.proj_in = nn.Linear(self.embed_dim, self.mag_hidden_dims[0])


    def forward(self, atoms, tags, pos):
        padding_mask = atoms.eq(0)
        n_graph, n_node = atoms.size()

        edge_type = atoms.view(n_graph, n_node, 1) * self.atom_types + atoms.view(n_graph, 1, n_node)

        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1)
        delta_pos /= dist.unsqueeze(-1) + 1e-5

        gbf_feature = self.gbf(dist, edge_type)

        edge_features = gbf_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1), 0.0
        )

        graph_node_feature = (
            self.tag_encoder(tags)
            + self.atom_encoder(atoms)
            + self.edge_proj(edge_features.sum(dim=-2))
        )

        non_zero_rows = torch.any(pos != 0, dim=-1)
    
        batch = torch.zeros(pos.view(-1, 3).shape[0], device=pos.device, dtype=torch.long)
        offsets = torch.arange(pos.shape[0], dtype=torch.long, device=pos.device) * pos.shape[1]

        batch[offsets] = 1
        batch = torch.cumsum(batch, 0) - 1

        pos_grouped = pos[non_zero_rows]
        batch_grouped = batch[non_zero_rows.flatten()]

        edge_index_radius = torch_geometric.nn.radius_graph(
            pos_grouped,
            self.cutoff_dist,
            batch_grouped,
            False,
            max_num_neighbors=self.num_nn,
            num_workers=0,
        )

        pad_diff = nearest_multiple_of_8(pos.shape[1]) - pos.shape[1]
        graph_node_feature = F.pad(graph_node_feature, pad=(0, 0, 0, pad_diff, 0, 0))

        h = self.st_fast(
            self.proj_in(graph_node_feature), edge_index_radius, batch_grouped, nearest_multiple_of_8(pos.shape[1])
        )

        predictions = torch.flatten(self.output_mlp(h.squeeze()))

        return predictions
    

    def configure_optimizers(self):
        opt = bnb.optim.AdamW8bit(self.parameters(), lr=self.lr, weight_decay=self.optimiser_weight_decay)

        self.monitor_loss_name = "Validation MCC" if "MCC" in self.monitor_loss_name or self.monitor_loss_name == "MCC" else self.monitor_loss_name
        mode = "max" if "MCC" in self.monitor_loss_name else "min"

        opt_dict = {
            "optimizer": opt,
            "monitor": self.monitor_loss_name,
        }

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode=mode, factor=0.5, patience=self.early_stopping_patience // 2, verbose=True
        )
        if self.monitor_loss_name != "train_loss":
            opt_dict["lr_scheduler"] = sched

        return opt_dict
    

    def _batch_loss(
        self,
        atoms: torch.Tensor,
        tags: torch.Tensor,
        pos: torch.Tensor,
        y: Optional[torch.Tensor],
        step_type: Optional[str] = None,
    ):
        predictions = self.forward(atoms, tags, pos)

        if self.task_type == "multi_classification":
            predictions = predictions.reshape(-1, self.linear_output_size)

            # Graph-level task
            if self.train_mask is None:
                task_loss = F.cross_entropy(predictions.squeeze().float(), y.squeeze().long())
            # Node-level task
            else:
                predictions = predictions.squeeze().float()
                y = y.squeeze().long()

                if step_type == "train" and self.train_mask is not None:
                    predictions = predictions[self.train_mask]
                    y = y[self.train_mask]
                
                if step_type == "validation" and self.val_mask is not None:
                    predictions = predictions[self.val_mask]
                    y = y[self.val_mask]

                if "test" in step_type and self.test_mask is not None:
                    predictions = predictions[self.test_mask]
                    y = y[self.test_mask]

                task_loss = F.cross_entropy(predictions, y)

        elif self.task_type == "binary_classification":
            y = y.view(predictions.shape)
            task_loss = F.binary_cross_entropy_with_logits(predictions.float(), y.float())

        else:
            if self.regression_loss_fn == "mse":
                task_loss = F.mse_loss(torch.flatten(predictions), torch.flatten(y.float()))
            elif self.regression_loss_fn == "mae":
                task_loss = F.l1_loss(torch.flatten(predictions), torch.flatten(y.float()))

        return task_loss, predictions, y


    def _step(self, batch: torch.Tensor, step_type: str):
        assert step_type in ["train", "validation", "test", "validation_test"]

        pos = batch["net_input"]["pos"]
        atoms = batch["net_input"]["atoms"]
        tags = batch["net_input"]["tags"]
        y = batch["targets"]["relaxed_energy"]

        task_loss, predictions = self._batch_loss(atoms, tags, pos, y)

        predictions = predictions.detach().squeeze()

        if self.task_type == "regression":
            output = (predictions.cpu(), y.cpu())
        elif "classification" in self.task_type:
            output = (predictions.cpu(), y.cpu())

        if step_type == "train":
            self.train_output[self.current_epoch].append(output)
        elif step_type == "validation":
            self.val_output[self.current_epoch].append(output)
        elif step_type == "validation_test":
            self.val_test_output[self.current_epoch].append(output)
        elif step_type == "test":
            self.test_output[self.num_called_test].append(output)

        return task_loss


    def training_step(self, batch: torch.Tensor, batch_idx: int):
        train_total_loss = self._step(batch, "train")

        if train_total_loss:
            self.log("train_loss", train_total_loss, prog_bar=True)

        return train_total_loss


    def validation_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            val_total_loss = self._step(batch, "validation")

            self.log("val_loss", val_total_loss)

            return val_total_loss

        if dataloader_idx == 1:
            val_test_total_loss = self._step(batch, "validation_test")

            self.log("val_test_loss", val_test_total_loss)

            return val_test_total_loss


    def test_step(self, batch: torch.Tensor, batch_idx: int):
        test_total_loss = self._step(batch, "test")

        self.log("test_loss", test_total_loss)

        return test_total_loss


    def _epoch_end_report(self, epoch_outputs, epoch_type):
        def flatten_list_of_tensors(lst):
            try:
                return np.array([item.item() for sublist in lst for item in sublist])
            except:
                return torch.cat(lst, dim=0)

        if self.task_type == "regression":
            y_pred = flatten_list_of_tensors([item[0] for item in epoch_outputs]).reshape(-1, self.linear_output_size)
            y_true = flatten_list_of_tensors([item[1] for item in epoch_outputs]).reshape(-1, self.linear_output_size)
        else:
            if self.batch_size > 1:
                y_pred = torch.cat([item[0] for item in epoch_outputs], dim=0)
                y_true = torch.cat([item[1] for item in epoch_outputs], dim=0)
            else:
                y_pred = torch.cat([item[0].unsqueeze(0) for item in epoch_outputs], dim=0).squeeze()
                y_true = torch.cat([item[1].unsqueeze(0) for item in epoch_outputs], dim=0).squeeze()

        if self.scaler:
            if self.linear_output_size > 1:
                y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, self.linear_output_size))
                y_true = self.scaler.inverse_transform(y_true.reshape(-1, self.linear_output_size))
            else:
                y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_true = self.scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

            y_pred = torch.from_numpy(y_pred)
            y_true = torch.from_numpy(y_true)

        if self.task_type == "binary_classification" and self.linear_output_size > 1:
            y_true = y_true.detach().cpu().reshape(-1, self.linear_output_size).long()
            y_pred = y_pred.detach().cpu().reshape(-1, self.linear_output_size)

            metrics = get_cls_metrics_multilabel_pt(y_true, y_pred, self.linear_output_size)

            self.log(f"{epoch_type} AUROC", metrics[0], batch_size=self.batch_size)
            self.log(f"{epoch_type} MCC", metrics[1], batch_size=self.batch_size)
            self.log(f"{epoch_type} Accuracy", metrics[2], batch_size=self.batch_size)
            self.log(f"{epoch_type} F1", metrics[3], batch_size=self.batch_size)

        elif self.task_type == "binary_classification" and self.linear_output_size == 1:
            metrics = get_cls_metrics_binary_pt(y_true, y_pred)

            self.log(f"{epoch_type} AUROC", metrics[0], batch_size=self.batch_size)
            self.log(f"{epoch_type} MCC", metrics[1], batch_size=self.batch_size)
            self.log(f"{epoch_type} Accuracy", metrics[2], batch_size=self.batch_size)
            self.log(f"{epoch_type} F1", metrics[3], batch_size=self.batch_size)

        elif self.task_type == "multi_classification" and self.linear_output_size > 1:
            metrics = get_cls_metrics_multiclass_pt(y_true, y_pred, self.linear_output_size)

            self.log(f"{epoch_type} AUROC", metrics[0], batch_size=self.batch_size)
            self.log(f"{epoch_type} MCC", metrics[1], batch_size=self.batch_size)
            self.log(f"{epoch_type} Accuracy", metrics[2], batch_size=self.batch_size)
            self.log(f"{epoch_type} F1", metrics[3], batch_size=self.batch_size)
            self.log(f"{epoch_type} AP", metrics[4], batch_size=self.batch_size)

        elif self.task_type == "regression":
            if self.linear_output_size != 11:
                metrics = get_regr_metrics_pt(y_true.squeeze(), y_pred.squeeze())

                self.log(f"{epoch_type} R2", metrics["R2"], batch_size=self.batch_size)
                self.log(f"{epoch_type} MAE", metrics["MAE"], batch_size=self.batch_size)
                self.log(f"{epoch_type} RMSE", metrics["RMSE"], batch_size=self.batch_size)
                self.log(f"{epoch_type} SMAPE", metrics["SMAPE"], batch_size=self.batch_size)
            else:
                mae_avg = []
                for idx in range(11):
                    target_name = pept_struct_target_names[idx]
                    metrics = get_regr_metrics_pt(y_true.squeeze()[:, idx], y_pred.squeeze()[:, idx])

                    mae_avg.append(metrics["MAE"])

                    self.log(f"{epoch_type} {target_name} R2", metrics["R2"], batch_size=self.batch_size)
                    self.log(f"{epoch_type} {target_name} MAE", metrics["MAE"], batch_size=self.batch_size)
                    self.log(f"{epoch_type} {target_name} RMSE", metrics["RMSE"], batch_size=self.batch_size)
                    self.log(f"{epoch_type} {target_name} SMAPE", metrics["SMAPE"], batch_size=self.batch_size)

                mae_avg = np.mean(np.array(mae_avg))
                self.log(f"{epoch_type} AVERAGE MAE", mae_avg, batch_size=self.batch_size)


        return metrics, y_pred, y_true


    def on_train_epoch_end(self):
        self.train_metrics[self.current_epoch], y_pred, y_true = self._epoch_end_report(
            self.train_output[self.current_epoch], epoch_type="Train"
        )

        del y_pred
        del y_true
        del self.train_output[self.current_epoch]


    def on_validation_epoch_end(self):
        if len(self.val_output[self.current_epoch]) > 0:
            self.val_metrics[self.current_epoch], y_pred, y_true = self._epoch_end_report(
                self.val_output[self.current_epoch], epoch_type="Validation"
            )

            del y_pred
            del y_true
            del self.val_output[self.current_epoch]

        if len(self.val_test_output[self.current_epoch]) > 0:
            self.val_test_metrics[self.current_epoch], y_pred, y_true = self._epoch_end_report(
                self.val_test_output[self.current_epoch], epoch_type="ValidationTest"
            )

            del y_pred
            del y_true
            del self.val_test_output[self.current_epoch]


    def on_test_epoch_end(self):
        test_outputs_per_epoch = self.test_output[self.num_called_test]
        self.test_metrics[self.num_called_test], y_pred, y_true = self._epoch_end_report(
            test_outputs_per_epoch, epoch_type="Test"
        )
        self.test_output[self.num_called_test] = y_pred
        self.test_true[self.num_called_test] = y_true

        self.num_called_test += 1
