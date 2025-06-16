import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """FiLM layers adapted from Perez et al., 2017"""

    def __init__(self, n_dims, wt_decay=1e-5):
        """
        :param n_dims: dimension of the inputs (2D/3D)
        :param wt_decay: L2 penalty on FiLM projection.
        """
        super(FiLM, self).__init__()
        self.n_dims = n_dims
        self.wt_decay = wt_decay
        self.channels = None
        self.fc = None

    def build(self, input_shape):
        self.channels = input_shape[0][1]  # input_shape: [x, z].
        self.init_channel = input_shape[1][1]
        self.fc = nn.Linear(self.init_channel, int(2 * self.channels))
        if self.wt_decay > 0:
            self.fc.weight = nn.Parameter(
                (self.fc.weight + self.wt_decay * torch.sum(self.fc.weight ** 2)).cuda()
            )
            self.fc.bias = nn.Parameter(
                (self.fc.bias + self.wt_decay * torch.sum(self.fc.bias ** 2)).cuda()
            )

    def forward(self, x, z):
        if self.fc is None:
            self.build((x.shape, z.shape))
        z = self.fc(z)
        gamma = z[..., :self.channels]
        beta = z[..., self.channels:]
        for _ in range(self.n_dims):
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        return (1. + gamma) * x + beta


class L2Regularizer(nn.Module):
    """re-implement this here for backwards compatibility with earlier versions of Keras"""

    def __init__(self, l2):
        super(L2Regularizer, self).__init__()
        self.l2 = l2

    def forward(self, x):
        return self.l2 * torch.sum(x ** 2)
