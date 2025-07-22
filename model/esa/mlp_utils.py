from torch import nn
from torch.nn import functional as F
from flash_attn.ops.activations import swiglu


class SmallMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        inter_dim,
        out_dim,
        dropout_p=0.0,
        num_layers=2,
        use_ln=False,
    ):
        super().__init__()

        self.mlp = []

        if num_layers == 1:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
                nn.Mish(),
            )
        else:
            for i in range(num_layers):
                if i == 0:
                    self.mlp.append(nn.Linear(in_dim, inter_dim))
                    if use_ln:
                        self.mlp.append(nn.LayerNorm(inter_dim))
                    self.mlp.append(nn.Mish())
                elif i != num_layers - 1:
                    self.mlp.append(nn.Linear(inter_dim, inter_dim))
                    if use_ln:
                        self.mlp.append(nn.LayerNorm(inter_dim))
                    self.mlp.append(nn.Mish())
                else:
                    self.mlp.append(nn.Linear(inter_dim, out_dim))

                if dropout_p > 0:
                    self.mlp.append(nn.Dropout(p=dropout_p))

            self.mlp = nn.Sequential(*self.mlp)


    def forward(self, x):
        return self.mlp(x)
    

class GatedMLPSingle(nn.Module):
    def __init__(
        self,
        in_dim,
        inter_dim,
        out_dim,
        dropout_p=0.0,
        use_ln=False,
    ):
        super().__init__()

        # Uncomment if you want dropout here
        # self.dropout_p = dropout_p

        self.fc1 = nn.Linear(in_dim, 2 * inter_dim, bias=True)
        self.fc2 = nn.Linear(inter_dim, out_dim, bias=True)
        self.use_ln = use_ln

        if self.use_ln:
            self.ln = nn.LayerNorm(2 * inter_dim, eps=1e-8)

        # if dropout_p > 0:
        #     self.dropout = nn.Dropout(p=dropout_p)


    def forward(self, x):
        if self.use_ln:
            y = self.ln(self.fc1(x))
        else:
            y = self.fc1(x)

        y, gate = y.chunk(2, dim=-1)
        y = swiglu(gate, y)

        # if self.dropout_p > 0:
        #     y = self.dropout(y)
        y = self.fc2(y)
        
        return y
    

class GatedMLPMulti(nn.Module):
    def __init__(
        self,
        in_dim,
        inter_dim,
        out_dim,
        dropout_p=0.0,
        num_layers=2,
        use_ln=False,
    ):
        super().__init__()

        self.mlp = []

        if num_layers == 1:
            self.mlp = nn.Sequential(
                GatedMLPSingle(in_dim, inter_dim, out_dim, dropout_p=dropout_p, use_ln=False),
                nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
                nn.Mish(),
            )
        else:
            for i in range(num_layers):
                if i == 0:
                    self.mlp.append(GatedMLPSingle(in_dim, inter_dim, inter_dim, dropout_p=dropout_p, use_ln=use_ln))
                elif i != num_layers - 1:
                    self.mlp.append(GatedMLPSingle(inter_dim, inter_dim, inter_dim, dropout_p=dropout_p, use_ln=use_ln))
                else:
                    self.mlp.append(GatedMLPSingle(inter_dim, inter_dim, out_dim, dropout_p=dropout_p, use_ln=use_ln))
                
                if dropout_p > 0:
                    self.mlp.append(nn.Dropout(p=dropout_p))

                self.mlp.append(nn.Mish())

        self.mlp = nn.Sequential(*self.mlp)


    def forward(self, x):
        return self.mlp(x)