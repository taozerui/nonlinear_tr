import torch
from torch import nn


class Linear(nn.Module):
    """ Linear """
    def __init__(self, in_features, out_features,
                 activation=None, bn=False, bias=True):
        super(Linear, self).__init__()
        net = [nn.Linear(in_features, out_features, bias=bias)]
        if bn:
            net.append(nn.LayerNorm(out_features))
        if activation is not None:
            net.append(activation)

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class PermLinear(nn.Module):
    """ MyLinear """
    def __init__(self, in_size, out_size):
        super(PermLinear, self).__init__()
        self.net = nn.Linear(in_size, out_size)

    def forward(self, x):
        assert x.ndim == 3
        out = self.net(x.permute(0, 2, 1))
        return out.permute(0, 2, 1)

