import torch
import admin_torch

from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import unbatch_edge_index, to_dense_batch
from torch_geometric.data import Batch

from typing import Optional
from utils.norm_layers import BN, LN
from esa.mha import SAB, PMA
from esa.mlp_utils import SmallMLP, GatedMLPMulti


def get_adj_mask_from_edge_index_node(
    edge_index, batch_size, max_items, batch_mapping, xformers_or_torch_attn, dtype, device="cuda:0"
):
    if xformers_or_torch_attn in ["torch"]:
        empty_mask_fill_value = False
        mask_dtype = torch.bool
        edge_mask_fill_value = True
    else:
        empty_mask_fill_value = -99999
        mask_dtype = dtype # 使用传入的dtype
        edge_mask_fill_value = 0

    adj_mask = torch.full(
        size=(batch_size, max_items, max_items),
        fill_value=empty_mask_fill_value,
        device=device,
        dtype=mask_dtype,
        requires_grad=False,
    )

    edge_index_unbatched = unbatch_edge_index(edge_index, batch_mapping)
    edge_batch_non_cumulative = torch.cat(edge_index_unbatched, dim=1)

    edge_batch_mapping = batch_mapping.index_select(0, edge_index[0, :])

    adj_mask[
        edge_batch_mapping, edge_batch_non_cumulative[0, :], edge_batch_non_cumulative[1, :]
    ] = edge_mask_fill_value

    if xformers_or_torch_attn in ["torch"]:
        adj_mask = ~adj_mask

    adj_mask = adj_mask.unsqueeze(1)
    return adj_mask


def create_edge_adjacency_mask(edge_index, num_edges):
    # Get all the nodes in the edge index (source and target separately for undirected graphs)
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]

    # Create expanded versions of the source and target node tensors
    expanded_source_nodes = source_nodes.unsqueeze(1).expand(-1, num_edges)
    expanded_target_nodes = target_nodes.unsqueeze(1).expand(-1, num_edges)

    # Create the adjacency mask where an edge is adjacent if either node matches either node of other edges
    source_adjacency = expanded_source_nodes == expanded_source_nodes.t()
    target_adjacency = expanded_target_nodes == expanded_target_nodes.t()
    cross_adjacency = (expanded_source_nodes == expanded_target_nodes.t()) | (
        expanded_target_nodes == expanded_source_nodes.t()
    )

    adjacency_mask = source_adjacency | target_adjacency | cross_adjacency

    # Mask out self-adjacency by setting the diagonal to False
    adjacency_mask.fill_diagonal_(0)  # We use "0" here to indicate False in PyTorch boolean context

    return adjacency_mask


def get_first_unique_index(t):
    # This is taken from Stack Overflow :)
    unique, idx, counts = torch.unique(t, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)

    zero = torch.tensor([0], device=t.device)
    cum_sum = torch.cat((zero, cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]

    return first_indicies


def generate_consecutive_tensor(input_tensor, final):
    # Calculate the length of each segment
    lengths = input_tensor[1:] - input_tensor[:-1]

    # Append the final length
    lengths = torch.cat((lengths, torch.tensor([final - input_tensor[-1]], device=input_tensor.device)))

    # Create ranges for each segment
    ranges = [torch.arange(0, length.item(), device=input_tensor.device) for length in lengths]

    # Concatenate all ranges into a single tensor
    if not ranges:
        return torch.empty(0, dtype=torch.long, device=input_tensor.device)
    return torch.cat(ranges)

# This is needed if the standard "nonzero" method from PyTorch fails
# This alternative is slower but allows bypassing the problem until 64-bit
# support is available
def nonzero_chunked(ten, num_chunks):
    # This is taken from this pull request
    # https://github.com/facebookresearch/segment-anything/pull/569/files
    b, w_h = ten.shape
    total_elements = b * w_h

    # Maximum allowable elements in one chunk - as torch is using 32 bit integers for this function
    max_elements_per_chunk = 2**31 - 1

    # Calculate the number of chunks needed
    if total_elements % max_elements_per_chunk != 0:
        num_chunks += 1

    # Calculate the actual chunk size
    chunk_size = b // num_chunks
    if b % num_chunks != 0:
        chunk_size += 1

    # List to store the results from each chunk
    all_indices = []

    # Loop through the diff tensor in chunks
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, b)
        chunk = ten[start:end, :]

        # Get non-zero indices for the current chunk
        indices = chunk.nonzero()

        # Adjust the row indices to the original tensor
        indices[:, 0] += start

        all_indices.append(indices)

    # Concatenate all the results
    change_indices = torch.cat(all_indices, dim=0)

    return change_indices


def get_adj_mask_from_edge_index_edge(
    edge_index,
    batch_size,
    max_items,
    batch_mapping,
    xformers_or_torch_attn,
    dtype,
    device="cuda:0",
):
    if xformers_or_torch_attn in ["torch"]:
        empty_mask_fill_value = False
        mask_dtype = torch.bool
        edge_mask_fill_value = True
    else:
        empty_mask_fill_value = -99999
        mask_dtype = dtype # 使用传入的dtype
        edge_mask_fill_value = 0

    adj_mask = torch.full(
        size=(batch_size, max_items, max_items),
        fill_value=empty_mask_fill_value,
        device=device,
        dtype=mask_dtype,
        requires_grad=False,
    )

    edge_batch_mapping = batch_mapping.index_select(0, edge_index[0, :])

    edge_adj_matrix = create_edge_adjacency_mask(edge_index, edge_index.shape[1])

    edge_batch_index_to_original_index = generate_consecutive_tensor(
        get_first_unique_index(edge_batch_mapping), edge_batch_mapping.shape[0]
    )

    try:
        eam_nonzero = edge_adj_matrix.nonzero()
    except:
        # Adjust chunk size as desired
        eam_nonzero = nonzero_chunked(edge_adj_matrix, 3)

    adj_mask[
        edge_batch_mapping.index_select(0, eam_nonzero[:, 0]),
        edge_batch_index_to_original_index.index_select(0, eam_nonzero[:, 0]),
        edge_batch_index_to_original_index.index_select(0, eam_nonzero[:, 1]),
    ] = edge_mask_fill_value


    if xformers_or_torch_attn in ["torch"]:
        adj_mask = ~adj_mask

    adj_mask = adj_mask.unsqueeze(1)
    return adj_mask


class SABComplete(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        dropout,
        idx,
        norm_type,
        use_mlp=False,
        mlp_hidden_size=64,
        mlp_type="standard",
        node_or_edge="edge",
        xformers_or_torch_attn="xformers",
        residual_dropout=0,
        set_max_items=0,
        use_bfloat16=True,
        num_mlp_layers=3,
        pre_or_post="pre",
        num_layers_for_residual=0,
        use_mlp_ln=False,
        mlp_dropout=0,
    ):
        super(SABComplete, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.use_mlp = use_mlp
        self.idx = idx
        self.mlp_hidden_size = mlp_hidden_size
        self.node_or_edge = node_or_edge
        self.xformers_or_torch_attn = xformers_or_torch_attn
        self.residual_dropout = residual_dropout
        self.set_max_items = set_max_items
        self.use_bfloat16 = use_bfloat16
        self.num_mlp_layers = num_mlp_layers
        self.pre_or_post = pre_or_post

        if self.pre_or_post == "post":
            self.residual_attn = admin_torch.as_module(num_layers_for_residual)
            self.residual_mlp = admin_torch.as_module(num_layers_for_residual)

        if dim_in != dim_out:
            self.proj_1 = nn.Linear(dim_in, dim_out)
    

        self.sab = SAB(dim_in, dim_out, num_heads, dropout, xformers_or_torch_attn)

        if self.idx != 2:
            bn_dim = self.set_max_items
        else:
            bn_dim = 32

        if norm_type == "LN":
            if self.pre_or_post == "post":
                if self.idx != 2:
                    self.norm = LN(dim_out, num_elements=self.set_max_items)
                else:
                    self.norm = LN(dim_out)
            else:
                if self.idx != 2:
                    self.norm = LN(dim_in, num_elements=self.set_max_items)
                else:
                    self.norm = LN(dim_in)
                    
        elif norm_type == "BN":
            self.norm = BN(bn_dim)

        self.mlp_type = mlp_type

        if self.use_mlp:
            if self.mlp_type == "standard":
                self.mlp = SmallMLP(
                    in_dim=dim_out,
                    out_dim=dim_out,
                    inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout,
                    num_layers=num_mlp_layers,
                    use_ln=use_mlp_ln,
                )
            
            elif self.mlp_type == "gated_mlp":
                self.mlp = GatedMLPMulti(
                    in_dim=dim_out,
                    out_dim=dim_out,
                    inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout,
                    num_layers=num_mlp_layers,
                    use_ln=use_mlp_ln,
                )

        if norm_type == "LN":
            if self.idx != 2:
                self.norm_mlp = LN(dim_out, num_elements=self.set_max_items)
            else:
                self.norm_mlp = LN(dim_out)
                
        elif norm_type == "BN":
            self.norm_mlp = BN(bn_dim)


    def forward(self, inp, return_attention=False):
        X, edge_index, batch_mapping, max_items, adj_mask = inp
        attn_weights = None

        if self.pre_or_post == "pre":
            X = self.norm(X)

        # --- Attention Block ---
        sab_input = X
        sab_mask = adj_mask if self.idx == 1 else None
        
        sab_out = self.sab(sab_input, sab_mask, return_attention=return_attention)
        
        if return_attention:
            out_attn, attn_weights = sab_out
        else:
            out_attn = sab_out

        if out_attn.shape[-1] != X.shape[-1]:
            X = self.proj_1(X)

        if self.pre_or_post == "pre":
            out = X + out_attn
        else: # post-norm
            out = self.residual_attn(X, out_attn)
            out = self.norm(out)
        
        # --- MLP Block ---
        if self.use_mlp:
            mlp_input = out
            if self.pre_or_post == "pre":
                out_mlp = self.norm_mlp(mlp_input)
                out_mlp = self.mlp(out_mlp)
                if mlp_input.shape[-1] == out_mlp.shape[-1]:
                    out = out_mlp + mlp_input
            else: # post-norm
                out_mlp = self.mlp(mlp_input)
                if mlp_input.shape[-1] == out_mlp.shape[-1]:
                    out = self.residual_mlp(mlp_input, out_mlp)
                out = self.norm_mlp(out)

        if self.residual_dropout > 0:
            out = F.dropout(out, p=self.residual_dropout)

        if return_attention:
            return (out, edge_index, batch_mapping, max_items, adj_mask), attn_weights
        return out, edge_index, batch_mapping, max_items, adj_mask


class PMAComplete(nn.Module):
    def __init__(
        self,
        dim_hidden,
        num_heads,
        num_outputs,
        norm_type,
        dropout=0,
        use_mlp=False,
        mlp_hidden_size=64,
        mlp_type="standard",
        xformers_or_torch_attn="xformers",
        set_max_items=0,
        use_bfloat16=True,
        num_mlp_layers=3,
        pre_or_post="pre",
        num_layers_for_residual=0,
        residual_dropout: float = 0.0,
        use_mlp_ln=False,
        mlp_dropout: float = 0.0,
    ):
        super(PMAComplete, self).__init__()

        self.use_mlp = use_mlp
        self.mlp_hidden_size = mlp_hidden_size
        self.num_heads = num_heads
        self.xformers_or_torch_attn = xformers_or_torch_attn
        self.set_max_items = set_max_items
        self.use_bfloat16 = use_bfloat16
        self.residual_dropout = residual_dropout
        self.num_mlp_layers = num_mlp_layers
        self.pre_or_post = pre_or_post

        if self.pre_or_post == "post":
            self.residual_attn = admin_torch.as_module(num_layers_for_residual)
            self.residual_mlp = admin_torch.as_module(num_layers_for_residual)

        self.pma = PMA(dim_hidden, num_heads, num_outputs, dropout, xformers_or_torch_attn)

        if norm_type == "LN":
            self.norm = LN(dim_hidden)
        elif norm_type == "BN":
            self.norm = BN(self.set_max_items)

        self.mlp_type = mlp_type

        if self.use_mlp:
            if self.mlp_type == "standard":
                self.mlp = SmallMLP(
                    in_dim=dim_hidden,
                    out_dim=dim_hidden,
                    inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout,
                    num_layers=num_mlp_layers,
                    use_ln=use_mlp_ln,
                )

            elif self.mlp_type == "gated_mlp":
                self.mlp = GatedMLPMulti(
                    in_dim=dim_hidden,
                    out_dim=dim_hidden,
                    inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout,
                    num_layers=num_mlp_layers,
                    use_ln=use_mlp_ln,
                )

        if norm_type == "LN":
            self.norm_mlp = LN(dim_hidden)
        elif norm_type == "BN":
            self.norm_mlp = BN(32)


    def forward(self, inp, return_attention=False):
        X, edge_index, batch_mapping, max_items, adj_mask = inp
        attn_weights = None

        if self.pre_or_post == "pre":
            X = self.norm(X)

        # --- Attention Block ---
        pma_out = self.pma(X, return_attention=return_attention)
        if return_attention:
            out_attn, attn_weights = pma_out
        else:
            out_attn = pma_out

        if self.pre_or_post == "pre" and out_attn.shape[-2] == X.shape[-2]:
            out = X + out_attn
        elif self.pre_or_post == "post" and out_attn.shape[-2] == X.shape[-2]:
            out = self.residual_attn(X, out_attn)
            out = self.norm(out)
        else:
            out = out_attn

        # --- MLP Block ---
        if self.use_mlp:
            mlp_input = out
            if self.pre_or_post == "pre":
                out_mlp = self.norm_mlp(mlp_input)
                out_mlp = self.mlp(out_mlp)
                if mlp_input.shape[-2] == out_mlp.shape[-2]:
                    out = out_mlp + mlp_input
            else: # post-norm
                out_mlp = self.mlp(mlp_input)
                if mlp_input.shape[-2] == out_mlp.shape[-2]:
                    out = self.residual_mlp(mlp_input, out_mlp)
                out = self.norm_mlp(out)

        if self.residual_dropout > 0:
            out = F.dropout(out, p=self.residual_dropout)

        if return_attention:
            return (out, edge_index, batch_mapping, max_items, adj_mask), attn_weights
        return out, edge_index, batch_mapping, max_items, adj_mask


class ESA(nn.Module):
    def __init__(
        self,
        num_outputs,
        dim_output,
        dim_hidden,
        num_heads,
        layer_types,
        node_or_edge="edge",
        xformers_or_torch_attn="xformers",
        pre_or_post="pre",
        norm_type="LN",
        sab_dropout=0.0,
        mab_dropout=0.0,
        pma_dropout=0.0,
        residual_dropout=0.0,
        pma_residual_dropout=0.0,
        use_mlps=False,
        mlp_hidden_size=64,
        num_mlp_layers=2,
        mlp_type="gated_mlp",
        mlp_dropout=0.0,
        use_mlp_ln=False,
        set_max_items=0,
        use_bfloat16=True,
    ):
        super(ESA, self).__init__()

        assert len(layer_types) == len(dim_hidden) and len(layer_types) == len(num_heads)

        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.layer_types = layer_types
        self.node_or_edge = node_or_edge
        self.xformers_or_torch_attn = xformers_or_torch_attn
        self.pre_or_post = pre_or_post
        self.norm_type = norm_type
        self.sab_dropout = sab_dropout
        self.mab_dropout = mab_dropout
        self.pma_dropout = pma_dropout
        self.residual_dropout = residual_dropout
        self.pma_residual_dropout = pma_residual_dropout
        self.use_mlps = use_mlps
        self.mlp_hidden_size = mlp_hidden_size
        self.num_mlp_layers = num_mlp_layers
        self.mlp_type = mlp_type
        self.mlp_dropout = mlp_dropout
        self.use_mlp_ln = use_mlp_ln
        self.set_max_items = set_max_items
        self.use_bfloat16 = use_bfloat16
        
        layer_tracker = 0

        self.encoder = []

        pma_encountered = False
        dim_pma = -1

        has_pma = "P" in self.layer_types

        for lt in self.layer_types:
            layer_in_dim = dim_hidden[layer_tracker]
            layer_num_heads = num_heads[layer_tracker]
            if lt != "P":
                if has_pma:
                    layer_out_dim = dim_hidden[layer_tracker + 1]
                else:
                    layer_out_dim = dim_hidden[layer_tracker]
            else:
                layer_out_dim = -1

            if lt == "S" and not pma_encountered:
                self.encoder.append(
                    SABComplete(
                        layer_in_dim,
                        layer_out_dim,
                        layer_num_heads,
                        idx=0,
                        dropout=sab_dropout,
                        node_or_edge=node_or_edge,
                        xformers_or_torch_attn=xformers_or_torch_attn,
                        pre_or_post=pre_or_post,
                        residual_dropout=residual_dropout,
                        use_mlp=use_mlps,
                        mlp_hidden_size=mlp_hidden_size,
                        mlp_dropout=mlp_dropout,
                        num_mlp_layers=num_mlp_layers,
                        use_mlp_ln=use_mlp_ln,
                        norm_type=norm_type,
                        mlp_type=mlp_type,
                        set_max_items=set_max_items,
                        use_bfloat16=use_bfloat16,
                        num_layers_for_residual=len(dim_hidden) * 2,
                    )
                )
                
                print(f"Added encoder SAB ({layer_in_dim}, {layer_out_dim}, {layer_num_heads})")

            if lt == "M" and not pma_encountered:
                self.encoder.append(
                    SABComplete(
                        layer_in_dim,
                        layer_out_dim,
                        layer_num_heads,
                        idx=1,
                        dropout=mab_dropout,
                        node_or_edge=node_or_edge,
                        xformers_or_torch_attn=xformers_or_torch_attn,
                        pre_or_post=pre_or_post,
                        residual_dropout=residual_dropout,
                        use_mlp=use_mlps,
                        mlp_hidden_size=mlp_hidden_size,
                        mlp_dropout=mlp_dropout,
                        num_mlp_layers=num_mlp_layers,
                        use_mlp_ln=use_mlp_ln,
                        norm_type=norm_type,
                        mlp_type=mlp_type,
                        set_max_items=set_max_items,
                        use_bfloat16=use_bfloat16,
                        num_layers_for_residual=len(dim_hidden) * 2,
                    )
                )
                
                print(f"Added encoder MAB ({layer_in_dim}, {layer_out_dim}, {layer_num_heads})")
                
            if lt == "P":
                pma_encountered = True
                dim_pma = layer_in_dim
                self.decoder = nn.ModuleList([
                    PMAComplete(
                        dim_hidden=layer_in_dim,
                        num_heads=layer_num_heads,
                        num_outputs=num_outputs,
                        norm_type=norm_type,
                        dropout=pma_dropout,
                        use_mlp=use_mlps,
                        mlp_hidden_size=mlp_hidden_size,
                        mlp_type=mlp_type,
                        xformers_or_torch_attn=xformers_or_torch_attn,
                        set_max_items=set_max_items,
                        use_bfloat16=use_bfloat16,
                        num_mlp_layers=num_mlp_layers,
                        pre_or_post=pre_or_post,
                        num_layers_for_residual=len(dim_hidden) * 2,
                        residual_dropout=pma_residual_dropout,
                        use_mlp_ln=use_mlp_ln,
                        mlp_dropout=mlp_dropout,
                    )
                ])

                print(f"Added decoder PMA ({layer_in_dim}, {layer_num_heads})")

            if lt == "S" and pma_encountered:
                self.decoder.append(
                    SABComplete(
                        layer_in_dim,
                        layer_out_dim,
                        layer_num_heads,
                        idx=2,
                        dropout=sab_dropout,
                        node_or_edge=node_or_edge,
                        xformers_or_torch_attn=xformers_or_torch_attn,
                        pre_or_post=pre_or_post,
                        residual_dropout=residual_dropout,
                        use_mlp=use_mlps,
                        mlp_hidden_size=mlp_hidden_size,
                        mlp_dropout=mlp_dropout,
                        num_mlp_layers=num_mlp_layers,
                        use_mlp_ln=use_mlp_ln,
                        norm_type=norm_type,
                        mlp_type=mlp_type,
                        set_max_items=set_max_items,
                        use_bfloat16=use_bfloat16,
                        num_layers_for_residual=len(dim_hidden) * 2,
                    )
                )

                print(f"Added decoder SAB ({layer_in_dim}, {layer_out_dim}, {layer_num_heads})")

            if lt != "P":
                layer_tracker += 1

        self.encoder = nn.Sequential(*self.encoder)
        if pma_encountered and hasattr(self, 'decoder'):
            self.decoder = nn.Sequential(*self.decoder)

        self.decoder_linear = nn.Linear(dim_hidden[-1], dim_output, bias=True)

        if has_pma and dim_hidden[0] != dim_pma:
            self.out_proj = nn.Linear(dim_hidden[0], dim_pma)

            self.dim_pma = dim_pma


    def forward(self, X, edge_index, batch_mapping, num_max_items, return_attention=False):
        attention_weights = {}
        adj_mask = None
        if self.node_or_edge == "node":
            adj_mask = get_adj_mask_from_edge_index_node(
                edge_index=edge_index,
                batch_mapping=batch_mapping,
                batch_size=X.shape[0],
                max_items=self.set_max_items,
                xformers_or_torch_attn=self.xformers_or_torch_attn,
                dtype=X.dtype, # 动态传递dtype
            )
        elif self.node_or_edge == "edge":
            adj_mask = get_adj_mask_from_edge_index_edge(
                edge_index=edge_index,
                batch_mapping=batch_mapping,
                batch_size=X.shape[0],
                max_items=self.set_max_items,
                xformers_or_torch_attn=self.xformers_or_torch_attn,
                dtype=X.dtype, # 动态传递dtype
            )

        # Pad the mask to match the padded input tensor X for hardware efficiency
        if adj_mask is not None:
            current_size = adj_mask.shape[-1]
            target_size = X.shape[1]

            if current_size < target_size:
                pad_amount = target_size - current_size
                padding = (0, pad_amount, 0, pad_amount)
                if self.xformers_or_torch_attn in ["torch"] or return_attention:
                    pad_value = True # For torch backend, mask is True for non-attending
                else:
                    pad_value = -99999
                adj_mask = F.pad(adj_mask, padding, "constant", value=pad_value)

        # --- Encoder ---
        inp = (X, edge_index, batch_mapping, num_max_items, adj_mask)
        for i, layer in enumerate(self.encoder):
            out = layer(inp, return_attention=return_attention)
            if return_attention:
                inp, attn = out
                attention_weights[f'encoder_layer_{i}'] = attn
            else:
                inp = out
        enc = inp[0]

        if hasattr(self, "dim_pma") and self.dim_hidden[0] != self.dim_pma:
            X = self.out_proj(X)

        enc = enc + X

        # --- Decoder ---
        if hasattr(self, "decoder"):
            inp = (enc, edge_index, batch_mapping, num_max_items, adj_mask)
            for i, layer in enumerate(self.decoder):
                out = layer(inp, return_attention=return_attention)
                if return_attention:
                    inp, attn = out
                    attention_weights[f'decoder_layer_{i}'] = attn
                else:
                    inp = out
            out = inp[0].mean(dim=1)
        else:
            # If no decoder, pool from encoder output
            if enc.dim() > 2:
                out = enc.mean(dim=1)
            else:
                out = enc

        final_prediction = F.mish(self.decoder_linear(out))
        
        if return_attention:
            return final_prediction, attention_weights
        return final_prediction

class Estimator(nn.Module):
    """
    An adapter class that wraps the complex ESA model with a simpler interface.
    It has a strict API and will raise a TypeError if unexpected configuration
    parameters are passed from the training script.
    """
    def __init__(
        self,
        # --- Parameters for Estimator's MLP ---
        graph_dim: int,
        num_features: int,
        edge_dim: int,
        # --- Parameters for ESA model ---
        hidden_dims: list,
        num_heads: list,
        layer_types: list,
        num_inds: int,
        set_max_items: int,
        linear_output_size: int = 1,
        use_fp16: bool = True,
        node_or_edge: str = "edge",
        xformers_or_torch_attn: str = "xformers",
        pre_or_post: str = "pre",
        norm_type: str = "LN",
        sab_dropout: float = 0.0,
        mab_dropout: float = 0.0,
        pma_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        pma_residual_dropout: float = 0.0,
        use_mlps: bool = False,
        mlp_hidden_size: int = 64,
        num_mlp_layers: int = 2,
        mlp_type: str = "gated_mlp",
        mlp_dropout: float = 0.0,
        use_mlp_ln: bool = False,
        # --- High-level config params that are now explicitly ignored ---
        # We accept them in the signature to prevent TypeErrors from the old config,
        # but they are not used inside the model itself.
        task_type: Optional[str] = None,
        monitor_loss_name: Optional[str] = None,
        regression_loss_fn: Optional[str] = None,
        posenc: Optional[str] = None,
        apply_attention_on: Optional[str] = None # To catch the old name
    ):
        super().__init__()
        # Store key parameters for access by other modules if needed
        self.hidden_dim = graph_dim
        self.num_inds = num_inds

        # The core attention model
        self.st_fast = ESA(
            num_outputs=num_inds,
            dim_output=linear_output_size,
            dim_hidden=hidden_dims,
            num_heads=num_heads,
            layer_types=layer_types,
            set_max_items=set_max_items,
            use_bfloat16=use_fp16,
            node_or_edge=node_or_edge,
            xformers_or_torch_attn=xformers_or_torch_attn,
            pre_or_post=pre_or_post,
            norm_type=norm_type,
            sab_dropout=sab_dropout,
            mab_dropout=mab_dropout,
            pma_dropout=pma_dropout,
            residual_dropout=residual_dropout,
            pma_residual_dropout=pma_residual_dropout,
            use_mlps=use_mlps,
            mlp_hidden_size=mlp_hidden_size,
            num_mlp_layers=num_mlp_layers,
            mlp_type=mlp_type,
            mlp_dropout=mlp_dropout,
            use_mlp_ln=use_mlp_ln,
        )

        # MLP to process combined node and edge features before attention
        self.node_edge_mlp = SmallMLP(
            in_dim=num_features * 2 + edge_dim,
            out_dim=graph_dim,
            inter_dim=graph_dim * 2,
            num_layers=2,
            dropout_p=0.1
        )

    def forward(self, batch: Batch, return_embeds: bool = False, return_attention: bool = False):
        """
        Handles the full graph processing pipeline.
        Can optionally return intermediate edge embeddings for multi-scale architectures,
        or attention weights for visualization.
        """
        x, edge_index, edge_attr, batch_mapping = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        
        # --- 边界情况处理：图中没有边 ---
        if edge_index is None or edge_index.shape[1] == 0:
            batch_size = int(torch.max(batch_mapping).item() + 1) if batch_mapping.numel() > 0 else 0
            if batch_size == 0:
                # Handle case where batch is empty
                if return_attention:
                    return torch.empty(0, self.st_fast.decoder_linear.out_features, device=x.device, dtype=x.dtype), {}
                return torch.empty(0, self.st_fast.decoder_linear.out_features, device=x.device, dtype=x.dtype)

            if return_attention:
                return torch.zeros(batch_size, self.st_fast.decoder_linear.out_features, device=x.device, dtype=x.dtype), {}
            return torch.zeros(batch_size, self.st_fast.decoder_linear.out_features, device=x.device, dtype=x.dtype)

        # 兼容不同版本的max_items获取方式
        if hasattr(batch, 'max_edge_global'):
            num_max_items = batch.max_edge_global.item()
        elif hasattr(batch, 'set_max_items'):
             num_max_items = batch.set_max_items + 1
        else:
            # Fallback if neither is present
            num_max_items = torch.max(batch.batch).item() + 1

        # 1. Create edge features from node features
        source = x[edge_index[0, :], :]
        target = x[edge_index[1, :], :]
        h = torch.cat((source, target), dim=1)

        if edge_attr is not None:
            h = torch.cat((h, edge_attr.float()), dim=1)

        # 2. Project edge features to the graph dimension
        h = self.node_edge_mlp(h)
        edge_batch_index = batch_mapping.index_select(0, edge_index[0, :])

        # --- Conditional return for multi-scale models ---
        if return_embeds:
            # For the atomic encoder, we return the processed edge embeddings
            # and the corresponding batch mapping for pooling.
            return h, edge_batch_index

        # 3. Convert to dense batch for the attention mechanism
        h, _ = to_dense_batch(h, edge_batch_index, fill_value=0, max_num_nodes=num_max_items)

        # 4. Pass through the core ESA model
        esa_out = self.st_fast(h, edge_index, batch_mapping, num_max_items=num_max_items, return_attention=return_attention)
        
        if return_attention:
            predictions, attn_weights = esa_out
            return predictions.squeeze(-1), attn_weights

        predictions = esa_out
        return predictions.squeeze(-1)
