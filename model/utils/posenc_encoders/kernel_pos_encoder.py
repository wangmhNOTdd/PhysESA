import torch
import torch.nn as nn

# Taken from https://github.com/davidbuterez/st-no-gnn-general-refactor/blob/540011477c606cee4436bdba0a9c30cd9f4fe586/code/Exphormer/graphgps/encoder/

class KernelPENodeEncoder(torch.nn.Module):
    """Configurable kernel-based Positional Encoding node encoder.

    The choice of which kernel-based statistics to use is configurable through
    setting of `kernel_type`. Based on this, the appropriate config is selected,
    and also the appropriate variable with precomputed kernel stats is then
    selected from PyG Data graphs in `forward` function.
    E.g., supported are 'RWSE', 'HKdiagSE', 'ElstaticSE'.

    PE of size `dim_pe` will get appended to each node feature vector.

    Args:
        dim_emb: Size of final node embedding
    """

    kernel_type = 'RWSE'  # Instantiated type of the KernelPE, e.g. RWSE

    def __init__(self):
        super().__init__()
        if self.kernel_type is None:
            raise ValueError(f"{self.__class__.__name__} has to be "
                             f"preconfigured by setting 'kernel_type' class"
                             f"variable before calling the constructor.")

        dim_pe = 24  # Size of the kernel-based PE embedding
        num_rw_steps = 20
        model_type = "linear"  # Encoder NN model type for PEs
        n_layers = 3  # Num. layers in PE encoder model
        norm_type = "batchnorm"  # Raw PE normalization layer type
        self.pass_as_var = True  # Pass PE also as a separate variable

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None

        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(num_rw_steps, dim_pe))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(num_rw_steps, 2 * dim_pe))
                layers.append(nn.ReLU())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(nn.ReLU())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(num_rw_steps, dim_pe)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, pos_enc):
        # pestat_var = f"pestat_{self.kernel_type}"
        # if not hasattr(batch, pestat_var):
        #     raise ValueError(f"Precomputed '{pestat_var}' variable is "
        #                      f"required for {self.__class__.__name__}; set "
        #                      f"config 'posenc_{self.kernel_type}.enable' to "
        #                      f"True, and also set 'posenc.kernel.times' values")

        # pos_enc = getattr(batch, pestat_var)  # (Num nodes) x (Num kernel times)
        # pos_enc = batch.rw_landing  # (Num nodes) x (Num kernel times)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        return pos_enc

        # # Expand node features if needed
        # if self.expand_x:
        #     h = self.linear_x(batch.x.to(torch.float32))
        # else:
        #     h = batch.x.to(torch.float32)
        # # Concatenate final PEs to input embedding
        # batch.x = torch.cat((h, pos_enc), 1)
        # # Keep PE also separate in a variable (e.g. for skip connections to input)
        # if self.pass_as_var:
        #     setattr(batch, f'pe_{self.kernel_type}', pos_enc)
        # return batch