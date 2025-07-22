import sys
import os
import warnings
import argparse
import copy
import torch
import wandb
import numpy as np
import pytorch_lightning as pl

from pathlib import Path
from torch_geometric.seed import seed_everything
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# Imports from this project
sys.path.append(os.path.realpath("."))

from esa.models import Estimator

from data_loading.data_loading import get_dataset_train_val_test
from esa.config import (
    save_arguments_to_json,
    load_arguments_from_json,
    validate_argparse_arguments,
    get_wandb_name,
)

warnings.filterwarnings("ignore")

os.environ["WANDB__SERVICE_WAIT"] = "500"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def check_is_node_level_dataset(dataset_name):
    if dataset_name in ["PPI", "Cora", "CiteSeer"]:
        return True
    elif "infected" in dataset_name:
        return True
    elif "hetero" in dataset_name:
        return True
    
    return False

def main():
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()

    # Seed for seed_everything
    parser.add_argument("--seed", type=int)

    # Dataset arguments
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--dataset-download-dir", type=str)
    parser.add_argument("--dataset-one-hot", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset-target-name", type=str)

    # Learning hyperparameters
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--norm-type", type=str, choices=["BN", "LN"])
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--monitor-loss-name", type=str)
    parser.add_argument("--gradient-clip-val", type=float, default=0.5)
    parser.add_argument("--optimiser-weight-decay", type=float, default=1e-3)
    parser.add_argument("--regression-loss-fn", type=str, choices=["mae", "mse"])
    parser.add_argument("--use-bfloat16", default=True, action=argparse.BooleanOptionalAction)

    # Node/graph dimensions
    parser.add_argument("--graph-dim", type=int)

    # ESA arguments
    parser.add_argument("--xformers-or-torch-attn", type=str, choices=["xformers", "torch"])
    parser.add_argument("--apply-attention-on", type=str, choices=["node", "edge"], default="edge")
    parser.add_argument("--layer-types", type=str, nargs="+")
    parser.add_argument("--hidden-dims", type=int, nargs="+")
    parser.add_argument("--num-heads", type=int, nargs="+")
    parser.add_argument("--pre-or-post", type=str, choices=["pre", "post"], default="post")
    parser.add_argument("--sab-dropout", type=float, default=0.0)
    parser.add_argument("--mab-dropout", type=float, default=0.0)
    parser.add_argument("--pma-dropout", type=float, default=0.0)
    parser.add_argument("--attn-residual-dropout", type=float, default=0.0)
    parser.add_argument("--pma-residual-dropout", type=float, default=0.0)
    parser.add_argument("--pos-enc", type=str)
    
    # MLP arguments
    parser.add_argument("--use-mlps", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--mlp-hidden-size", type=int, default=64)
    parser.add_argument("--mlp-layers", type=int, default=2)
    parser.add_argument("--mlp-type", type=str, choices=["standard", "gated_mlp"], default="gated_mlp")
    parser.add_argument("--mlp-dropout", type=float, default=0.0)
    parser.add_argument("--use-mlp-ln", type=str, choices=["yes", "no"], default="yes")

    # 3D OCP settings
    parser.add_argument("--ocp-num-kernels", type=int)
    parser.add_argument("--ocp-embed-dim", type=int)
    parser.add_argument("--ocp-cutoff-dist", type=float)
    parser.add_argument("--ocp-num-neigh", type=int)

    # Path/config arguments
    parser.add_argument("--ckpt-path", type=str)
    parser.add_argument("--out-path", type=str)
    parser.add_argument("--config-json-path", type=str)
    parser.add_argument("--wandb-project-name", type=str)
    

    args = parser.parse_args()

    if args.config_json_path:
        argsdict = load_arguments_from_json(args.config_json_path)
        validate_argparse_arguments(argsdict)
    else:
        argsdict = vars(args)
        validate_argparse_arguments(argsdict)
        del argsdict["config_json_path"]

    seed_everything(argsdict["seed"])

    # Dataset arguments
    dataset = argsdict["dataset"]
    dataset_download_dir = argsdict["dataset_download_dir"]
    dataset_one_hot = argsdict["dataset_one_hot"]
    target_name = argsdict["dataset_target_name"]

    # Learning hyperparameters
    batch_size = argsdict["batch_size"]
    early_stopping_patience = argsdict["early_stopping_patience"]
    gradient_clip_val = argsdict["gradient_clip_val"]
    use_bfloat16 = argsdict["use_bfloat16"]
    apply_attention_on = argsdict["apply_attention_on"]
    mlp_dropout = argsdict["mlp_dropout"]
    regr_fn = argsdict["regression_loss_fn"]

    # # 3D OCP settings
    num_kernels = argsdict["ocp_num_kernels"]
    embed_dim = argsdict["ocp_embed_dim"]
    cutoff_dist = argsdict["ocp_cutoff_dist"]
    num_nn = argsdict["ocp_num_neigh"]

    # Path/config arguments
    ckpt_path = argsdict["ckpt_path"]
    out_path = argsdict["out_path"]
    wandb_project_name = argsdict["wandb_project_name"]
    monitor_loss_name = argsdict["monitor_loss_name"]
    num_mlp_layers = argsdict["mlp_layers"]
    pre_or_post = argsdict["pre_or_post"]
    pma_residual_dropout = argsdict["pma_residual_dropout"]
    use_mlp_ln = argsdict["use_mlp_ln"] == "yes"

    if monitor_loss_name == "MCC" or "MCC" in monitor_loss_name:
        monitor_loss_name = "Validation MCC"

    if argsdict["pos_enc"]:
        posenc = argsdict["pos_enc"].split("+")
    else:
        posenc = []

    print(f"Using {posenc} PE!")
        
    if check_is_node_level_dataset(dataset):
        assert "P" not in argsdict["layer_types"]

    if check_is_node_level_dataset(dataset):
        assert apply_attention_on == "node", "ESA is not currently supported for node-level tasks!"

    if dataset == "ocp":
        assert apply_attention_on == "node", "ESA is not currently supported for OCP!"
        
    if dataset in ["ESOL", "FreeSolv", "Lipo", "QM9", "DOCKSTRING", "ZINC", "PCQM4Mv2", "lrgb-pept-struct"]:
        assert regr_fn is not None, "A loss functions must be specified for regression tasks!"

    if dataset in ["QM9", "DOCKSTRING"]:
        assert target_name is not None, "A target must be specified for QM9 and DOCKSTRING!"

    ############## Data loading ##############
    train_mask, val_mask, test_mask = None, None, None

    if dataset != "ocp":
        if check_is_node_level_dataset(dataset):
            # Node-level task branch
            train, val, test, num_classes, task_type, scaler, train_mask, val_mask, test_mask = get_dataset_train_val_test(
                dataset=dataset,
                dataset_dir=dataset_download_dir,
            )
        # Graph-level branch
        else:
            train, val, test, num_classes, task_type, scaler = get_dataset_train_val_test(
                dataset=dataset,
                dataset_dir=dataset_download_dir,
                one_hot=dataset_one_hot,
                target_name=target_name,
                pe_types=posenc,
            )

        num_features = train[0].x.shape[-1]
        edge_dim = None
        if hasattr(train[0], "edge_attr") and train[0].edge_attr is not None:
            edge_dim = train[0].edge_attr.shape[-1]

        train_loader = GeometricDataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = GeometricDataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = GeometricDataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # OCP branch
    else:
        # Warning: requires fairseq
        from data_loading.data_loading_OCP import load_ocp
        from esa.models_OCP import Estimator as Estimator_OCP
        print("Please download the corresponding data files from https://fair-chem.github.io/core/datasets/oc20.html")
        train, scaler = load_ocp("/media/david/Media/ocp_test_nov2024/is2re/10k/train/data.lmdb", is_train=True, scaler=None)
        val, _ = load_ocp("/media/david/Media/ocp_test_nov2024/is2re/all/val_id/data.lmdb", is_train=False, scaler=scaler)
        test, _ = load_ocp("/media/david/Media/ocp_test_nov2024/is2re/all/val_id/data.lmdb", is_train=False, scaler=scaler)

        def collate_fn_train(batch):
            return train.collater(batch)
        
        def collate_fn_val(batch):
            return val.collater(batch)
        
        def collate_fn_test(batch):
            return test.collater(batch)

        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, collate_fn=collate_fn_train)
        val_loader= DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate_fn_val)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate_fn_test)

        task_type = "regression"
        num_features, num_classes, edge_dim = None, 1, None
    ############## Data loading ##############

    run_name = get_wandb_name(argsdict)

    output_save_dir = os.path.join(out_path)
    Path(output_save_dir).mkdir(exist_ok=True, parents=True)

    config_json_path = save_arguments_to_json(argsdict, output_save_dir)

    # Logging
    logger = WandbLogger(name=run_name, project=wandb_project_name, save_dir=output_save_dir)

    # Callbacks
    monitor_mode = "max" if "MCC" in monitor_loss_name else "min"
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_loss_name,
        dirpath=output_save_dir,
        filename="{epoch:03d}",
        mode=monitor_mode,
        save_top_k=1,
    )

    early_stopping_callback = EarlyStopping(
        monitor=monitor_loss_name, patience=early_stopping_patience, mode=monitor_mode
    )

    ############## Learning and model set-up ##############
    model_args = copy.deepcopy(argsdict)
    set_max_items = None

    if dataset != "ocp":
        set_max_items = train[0].max_edge_global if apply_attention_on == "edge" else train[0].max_node_global

    model_args = model_args | dict(
        task_type=task_type, num_features=num_features, linear_output_size=num_classes, scaler=scaler, edge_dim=edge_dim,
        set_max_items=set_max_items, pma_residual_dropout=pma_residual_dropout,
        k=num_kernels, embed_dim=embed_dim, cutoff_dist=cutoff_dist, num_nn=num_nn,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        posenc=posenc, num_mlp_layers=num_mlp_layers, pre_or_post=pre_or_post,
        use_mlp_ln=use_mlp_ln, mlp_dropout=mlp_dropout,
    )

    if dataset != "ocp":
        model_args |= dict(is_node_task=check_is_node_level_dataset(dataset))

    model = Estimator(**model_args) if dataset != "ocp" else Estimator_OCP(**model_args)
    model = model.cuda()

    trainer_args = dict(
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        min_epochs=2,
        max_epochs=-1,
        devices=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        precision="bf16" if use_bfloat16 else "32",
        gradient_clip_val=gradient_clip_val,
    )

    trainer_args = trainer_args | dict(accelerator="gpu")

    ############## Learning and model set-up ##############
    trainer = pl.Trainer(**trainer_args)

    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=[val_loader, test_loader], ckpt_path=ckpt_path
    )
    trainer.test(model=model, dataloaders=test_loader, ckpt_path="best")

    # Save test metrics
    preds_path = os.path.join(output_save_dir, "test_y_pred.npy")
    true_path = os.path.join(output_save_dir, "test_y_true.npy")
    metrics_path = os.path.join(output_save_dir, "test_metrics.npy")

    np.save(preds_path, model.test_output)
    np.save(true_path, model.test_true)
    np.save(metrics_path, model.test_metrics)

    wandb.save(preds_path)
    wandb.save(true_path)
    wandb.save(metrics_path)
    wandb.save(config_json_path)

    # ckpt_paths = [str(p) for p in Path(output_save_dir).rglob("*.ckpt")]
    # for cp in ckpt_paths:
    #     wandb.save(cp)

    wandb.finish()


if __name__ == "__main__":
    main()
