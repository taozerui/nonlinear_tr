import numpy as np
import torch
from torch import nn
from typing import Union, List

# from .tensor import near_eye_init
from .nn import Linear


class TensorLayer(nn.Module):
    """ TRC """
    def __init__(self, shape, hidden, act=nn.ELU()):
        super(TensorLayer, self).__init__()
        self.shape = shape
        self.dim = len(shape)
        self.act = act
        if isinstance(hidden, int):
            self.hidden = [hidden] * self.dim
        else:
            assert len(hidden) == self.dim
            self.hidden = hidden

        weight = []
        for d in range(self.dim):
            weight.append(nn.Parameter(
                torch.empty(shape[d], self.hidden[d]), requires_grad=True
            ))
            nn.init.xavier_normal_(weight[-1], 0.2)
        self.weight = nn.ParameterList(weight)

    def forward(self, x):
        batch_dim = x.ndim - self.dim
        for d in range(self.dim):
            x = torch.tensordot(x, self.weight[d], dims=([batch_dim], [0]))

        if self.act is not None:
            x = self.act(x)

        return x


class MLPEncoderDecoder(nn.Module):
    """ MLPEncoder """
    def __init__(self,
                 input_size: int,
                 output_size: List[int],
                 hidden_size: List[int],
                 act: Union[nn.Module, None],
                 out_act: Union[list, nn.Module, None],
                 bn: bool):
        super(MLPEncoderDecoder, self).__init__()
        if type(out_act) is list:
            assert len(output_size) == len(out_act)

        self.input_size = input_size
        self.latent_size = output_size

        layer_size = [input_size] + hidden_size
        if len(output_size) == 1:
            layers = nn.Sequential()
            for i in range(len(layer_size)):
                if i == len(layer_size) - 1:  # last layer
                    if type(out_act) is list:
                        layers.add_module(
                            f'Layer {i}', Linear(layer_size[i], output_size[0], bn=bn, activation=out_act[0])
                        )
                    else:
                        layers.add_module(
                            f'Layer {i}', Linear(layer_size[i], output_size[0], bn=bn, activation=out_act)
                        )
                else:
                    layers.add_module(
                        f'Layer {i}', Linear(layer_size[i], layer_size[i + 1], bn=bn, activation=act)
                    )
        else:
            layers = []
            for d in range(len(output_size)):
                ld = nn.Sequential()
                for i in range(len(layer_size)):
                    if i == len(layer_size) - 1:
                        if type(out_act) is list:
                            ld.add_module(
                                f'Layer {i}', Linear(layer_size[i], output_size[0], bn=bn, activation=out_act[d])
                            )
                        else:
                            ld.add_module(
                                f'Layer {i}', Linear(layer_size[i], output_size[0], bn=bn, activation=out_act)
                            )
                    else:
                        ld.add_module(
                            f'Layer {i}', Linear(layer_size[i], layer_size[i + 1], bn=bn, activation=act)
                        )
                layers.append(ld)
            layers = nn.ModuleList(layers)
        self.layers = layers

    def forward(self, x):
        if len(self.latent_size) == 1:
            out = self.layers(x)
            return out
        else:
            out = []
            for layer in self.layers:
                out.append(layer(x))
            return out


class MLPEncoderDecoderList(nn.Module):
    """ MLPEncoder """
    def __init__(self,
                 input_size: List[int],
                 output_size: List[int],
                 hidden_size: List[int],
                 act: Union[nn.Module, None],
                 out_act: Union[list, nn.Module, None],
                 bn: bool):
        super(MLPEncoderDecoderList, self).__init__()
        assert len(input_size) == len(output_size)
        if type(out_act) is list:
            assert len(output_size) == len(out_act)

        self.input_size = input_size
        self.latent_size = output_size

        layer_num = len(input_size)
        layers = []
        for i in range(layer_num):
            layer_size = [input_size[i]] + hidden_size
            layer_i = nn.Sequential()
            for l in range(len(layer_size)):
                if l == len(layer_size) - 1:  # last layer
                    if type(out_act) is list:
                        layer_i.add_module(
                            f'Net {i} Layer {l}', Linear(layer_size[l], output_size[i], bn=bn, activation=out_act[i])
                        )
                    else:
                        layer_i.add_module(
                            f'Net {i} Layer {l}', Linear(layer_size[l], output_size[i], bn=bn, activation=out_act)
                        )
                else:
                    layer_i.add_module(
                        f'Net {i} Layer {l}', Linear(layer_size[l], layer_size[l + 1], bn=bn, activation=act)
                    )
            layers.append(layer_i)

        self.layers = nn.ModuleList(layers)

    def forward(self, x: List):
        out = []
        k = 0
        for layer in self.layers:
            out.append(layer(x[k]))
            k += 1
        return out


class TensorContractionLayer(nn.Module):
    """ TensorContractionLayer """
    def __init__(self, shape, rank):
        super(TensorContractionLayer, self).__init__()
        dim = len(shape)
        self.dim = dim
        self.shape = shape
        self.rank = rank

        cores = []
        for d in range(dim):
            cores.append(nn.Parameter(torch.empty(shape[d], rank, rank)))
        self.cores = nn.ParameterList(cores)

        self.reset_parameters()

    def reset_parameters(self):
        for core in self.cores:
            nn.init.xavier_normal_(core, 0.2)
        # add bias?
        # if self.bias is not None:
        #     nn.init.xavier_normal_(self.bias, 0.5)

    def forward(self, x):
        assert x.ndim == self.dim + 1
        assert list(x.shape[1:]) == self.shape, "The shape of input imcompatible!"
        # encode each node
        latent_codes = []
        for d in range(self.dim):
            chain, index = self.subchain(self.cores, d)
            node = torch.tensordot(x, chain, dims=(index, list(range(1, self.dim))))
            latent_codes.append(node)

        return latent_codes

    def subchain(self, cores, skip):
        if skip == self.dim-1:
            out = cores[0]
            index = list(range(1, skip))
        else:
            out = cores[skip+1]
            index = list(range(skip+2, self.dim)) + list(range(0, skip))
        out = out.permute([1, 0, 2])
        for i in index:
            out = torch.tensordot(out, cores[i].permute([1, 0, 2]), dims=([-1], [0]))

        if skip == self.dim-1:
            index.insert(0, 0)
        else:
            index.insert(0, skip + 1)
        return out, [i+1 for i in index]  # plus 1 because we want to compute batch x
    
    
class SimpleTensorEncoder(nn.Module):
    """ SimpleTensorEncoder """
    def __init__(self, shape, rank, hidden=200, out_act=None):
        super(SimpleTensorEncoder, self).__init__()
        self.shape = shape
        self.dim = len(shape)
        net = []
        for d in range(self.dim):
            net.append(
                nn.Sequential(
                    Linear(np.prod(shape) // shape[d], hidden, bn=False, activation=nn.ELU()),
                    Linear(hidden, hidden, bn=False, activation=nn.ELU()),
                    Linear(hidden, rank ** 2, bn=False, activation=out_act),
                )
            )
        self.net = nn.ModuleList(net)

    def forward(self, x):
        assert x.ndim == self.dim + 1
        out = []
        for d in range(self.dim):
            # permute and reshape
            perm = list(range(1, self.dim + 1))
            perm.pop(d)
            perm = [0, d+1, *perm]

            x_ = x.permute(perm).view(-1, self.shape[d], np.prod(self.shape) // self.shape[d])
            out.append(self.net[d](x_))  # Batch x Id x R

        return out


if __name__ == '__main__':
    # net = MLPEncoderDecoderList(
    #     input_size=[10, 10, 10],
    #     output_size=[20, 30, 40],
    #     hidden_size=[15, 16],
    #     act=nn.ReLU(),
    #     out_act=nn.Sigmoid(),
    #     bn=True
    # )

    x = torch.rand(10, 3, 4)
    net = TensorContractionLayer(shape=[3, 4], rank=6)
    cores = net(x)
