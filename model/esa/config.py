import os
import json


def save_arguments_to_json(argsdict, out_path):
    json_out_path = os.path.join(out_path, "hyperparameters.json")
    with open(json_out_path, "w", encoding="UTF-8") as f:
        json.dump(argsdict, f)
    return json_out_path


def load_arguments_from_json(json_path):
    with open(json_path, "r", encoding="UTF-8") as f:
        argsdict = json.load(f)

    return argsdict


def validate_argparse_arguments(argsdict):
    assert "seed" in argsdict
    assert "dataset_download_dir" in argsdict
    assert "lr" in argsdict
    assert "batch_size" in argsdict
    assert "norm_type" in argsdict
    assert "early_stopping_patience" in argsdict
    assert "graph_dim" in argsdict
    assert "xformers_or_torch_attn" in argsdict
    assert "hidden_dims" in argsdict
    assert "num_heads" in argsdict
    assert "sab_dropout" in argsdict
    assert "mab_dropout" in argsdict
    assert "pma_dropout" in argsdict
    assert "apply_attention_on" in argsdict
    assert "use_mlps" in argsdict
    assert "out_path" in argsdict


def get_wandb_name(argsdict):
    name = ""

    if argsdict["apply_attention_on"] == "node":
        name = "NSA"
    elif argsdict["apply_attention_on"] == "edge":
        name = "ESA"

    name += f"+{''.join(argsdict['layer_types'])}"

    if "dataset" not in argsdict:
        dataset = "QM9-TF"
    else:
        dataset = argsdict['dataset']

    name += f"+{argsdict['xformers_or_torch_attn']}+{dataset}+S={argsdict['seed']}"

    if "dataset_one_hot" not in argsdict:
        dataset_one_hot = "yes"
    else:
        dataset_one_hot = "no"
    name += f"+OH={dataset_one_hot}"

    name += f"+T={argsdict['dataset_target_name']}"
    name += f"+NRM={argsdict['norm_type']}"
    name += f"+PorP={argsdict['pre_or_post']}"
    name += f"+BS={argsdict['batch_size']}+LR={argsdict['lr']}"
    name += f"+GDIM={argsdict['graph_dim']}"

    name += f"+GC={argsdict['gradient_clip_val']}"
    name += f"+OPTD={argsdict['optimiser_weight_decay']}"

    if "pos_enc" in argsdict and argsdict["pos_enc"]:
        name += f"+PE={argsdict['pos_enc'].replace('+', '-')}"
    else:
        name += "+PE=None"

    name += f"+MLP={argsdict['use_mlps']}"
    name += f"+MLPT={argsdict['mlp_type']}"
    name += f"+MLPL={argsdict['mlp_layers']}"
    name += f"+MLPN={argsdict['mlp_hidden_size']}"
    name += f"+MLPLN={argsdict['use_mlp_ln']}"
    name += f"+MLPD={argsdict['mlp_dropout']}"

    name += f"+RL={argsdict['regression_loss_fn']}"

    if len(argsdict["hidden_dims"]) > 1 and len(set(argsdict["hidden_dims"])) == 1:
        dim = argsdict["hidden_dims"][0]
    else:
        dim = argsdict["hidden_dims"]

    if len(argsdict["num_heads"]) > 1 and len(set(argsdict["num_heads"])) == 1:
        heads = argsdict["num_heads"][0]
    else:
        heads = argsdict["num_heads"]

    name += f"+DIM={dim}+HEADS={heads}"
    name += f"+SDR={argsdict['sab_dropout']}+MSDR={argsdict['mab_dropout']}"
    name += f"+PDR={argsdict['pma_dropout']}"

    name += f"+RDR={argsdict['attn_residual_dropout']}"
    name += f"+PRDR={argsdict['pma_residual_dropout']}"

    if "transfer_learning_hq_or_lq" in argsdict:
        hq_or_lq = argsdict["transfer_learning_hq_or_lq"]
        ind_or_trans = argsdict['transfer_learning_inductive_or_transductive']
        retrain_lq_to_hq = argsdict['transfer_learning_retrain_lq_to_hq'] == "yes"

        name += f"+TF-HQorLQ={hq_or_lq}"
        name += f"+TF-IorT={ind_or_trans}"
        name += f"+TF-tune={retrain_lq_to_hq}"
    
    return name
