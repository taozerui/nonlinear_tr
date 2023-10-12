# Tensor Decompositions

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from .tensor import full_batch_tr, unfold
from .layers import MLPEncoderDecoder, MLPEncoderDecoderList, TensorContractionLayer, SimpleTensorEncoder


class TR(nn.Module):
    """ TR """
    def __init__(self, args):
        super(TR, self).__init__()
        self.__name__ = 'AE-TR'
        self.args = args
        self.device = torch.device(args.device)
        self.input_len = int(np.prod(args.tensor_size))
        self.input_type = args.input_type

        self.loss = None
        self.beta = 0.0

        self.tensor_size = args.tensor_size
        self.tensor_dim = len(self.tensor_size)

        self._setup_encoder(args)
        self._setup_decoder(args)

    def _setup_encoder(self, args):
        encoder = []
        for d in range(self.tensor_dim):
            encoder.append(MLPEncoderDecoder(
                input_size=np.prod(self.tensor_size) // self.tensor_size[d],
                output_size=[args.in_rank ** 2],
                hidden_size=args.h_dim,
                act=nn.ELU(),
                out_act=nn.ELU(),
                bn=False
            ))
        self.encoder = nn.ModuleList(encoder)

    def _setup_decoder(self, args):
        h_decoder = args.h_dim.copy()
        h_decoder.reverse()
        self.decoder = MLPEncoderDecoderList(
            input_size=[args.in_rank ** 2] * self.tensor_dim,
            output_size=[args.out_rank ** 2] * self.tensor_dim,
            hidden_size=h_decoder,
            act=nn.ELU(),
            out_act=None,
            bn=False
        )

    def encode(self, x):
        if self.args.tensorize:
            x = x.view(-1, *self.tensor_size)

        cores = []
        for d in range(self.tensor_dim):
            x_mat = unfold(x, start_dim=1, dim=d+1)
            mu = self.encoder[d](x_mat)
            cores.append(mu)  # Batch x I x R
        return cores

    def decode(self, z):
        cores = self.decoder(z)
        for d in range(self.tensor_dim):
            cores[d] = cores[d].reshape(
                -1, self.tensor_size[d], self.args.out_rank, self.args.out_rank
            )
        mu = full_batch_tr(cores)
        mu = torch.sigmoid(mu)
        mu = mu.view(-1, *self.args.input_shape)
        return mu

    def forward(self, x, mask=None):
        cores = self.encode(x)
        x_hat = self.decode(cores)

        if self.input_type == 'binary':
            nll = F.binary_cross_entropy(x_hat, x, reduction='none')
        else:
            nll = (x_hat - x) ** 2

        if mask is not None:
            assert nll.shape == mask.shape
            nll = (nll * mask).sum() / x_hat.shape[0]
        else:
            nll = nll.sum() / x_hat.shape[0]
        self.loss = {  # mask the loss dict consistent with vae.
            'NELBO': nll, 'Recon': torch.zeros(1), 'KL': torch.zeros(1)
        }
        return x_hat  # Sample x Batch x Dim

    def impute(self, x, num=None):
        cores = self.encode(x)
        x_hat = self.decode(cores)
        return x_hat