from torch import nn
from utils.batch_renorm import BatchRenorm1d

class BatchRenorm(nn.Module):
    def __init__(self, dim, eps=1e-8, momentum=0.1):
        super(BatchRenorm, self).__init__()

        self.dim = dim

        self.bn = BatchRenorm1d(num_features=dim, eps=eps, momentum=momentum)

    def forward(self, x):
        return self.bn(x)


class BN(nn.Module):
    def __init__(self, dim, num_elements=None):
        super(BN, self).__init__()

        # self.bn = nn.BatchNorm1d(dim, eps=1e-8, momentum=0.1)
        self.bn = BatchRenorm1d(dim, eps=1e-8, momentum=0.1)

    def forward(self, x):
        return self.bn(x)


class LN(nn.Module):
    def __init__(self, hidden_dim, num_elements=None):
        super(LN, self).__init__()

        # if num_elements is not None:
        #     self.ln = nn.LayerNorm([num_elements, hidden_dim], eps=1e-8)
        # else:
        #     self.ln = nn.LayerNorm(hidden_dim, eps=1e-8)

        self.ln = nn.LayerNorm(hidden_dim, eps=1e-8)

    def forward(self, x):
        return self.ln(x)