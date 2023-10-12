import torch
import numpy as np
from torch import nn
from torch.distributions import Normal, Distribution
import pyro.contrib.gp as gp

from .tensor import full_batch_tr, unfold
from .layers import MLPEncoderDecoder, MLPEncoderDecoderList, SimpleTensorEncoder, TensorLayer
from .base import BaseVAE

from src.distribution import MatrixGP, kl_normal_gp, SparseMatrixGP


class VariationalTR(BaseVAE):
    """ VariationalTR """
    def __init__(self, args):
        super(VariationalTR, self).__init__(args)
        self.__name__ = 'TR-VAE'
        self.tensor_size = args.tensor_size
        self.tensor_dim = len(self.tensor_size)
        self._setup_prior(args)
        self._setup_encoder(args)
        self._setup_decoder(args)

        self._setup_conv_layers(args)

    def _setup_prior(self, args):
        if args.prior == 'normal':
            self.prior = Normal(
                loc=torch.zeros(args.in_rank ** 2).to(self.device),
                scale=torch.ones(args.in_rank ** 2).to(self.device)
            )
        elif args.prior == 'sparse_gp':
            self.prior = SparseMatrixGP(
                kernel=gp.kernels.RBF(input_dim=args.latent_feature),
                shape=self.tensor_size,
                pseudo_num=args.induce_num,
                tau=args.tau
            )
            self._setup_inducing_dist(args)
        else:
            raise NotImplementedError

    def _setup_inducing_dist(self, args):
        if args.induce_type == 'simple':
            q_m = []
            q_S = []
            for d in range(self.tensor_dim):
                q_m.append(nn.Parameter(
                    torch.empty(args.induce_num, self.tensor_size[d], args.in_rank ** 2), requires_grad=True
                ))
                q_S.append(nn.Parameter(
                    torch.empty(args.induce_num, self.tensor_size[d], args.in_rank ** 2), requires_grad=True
                ))
                nn.init.normal_(q_m[-1], std=1.0)
                nn.init.normal_(q_S[-1], std=1.0)
            self.q_m = nn.ParameterList(q_m)
            self.q_S = nn.ParameterList(q_S)
        elif args.induce_type == 'encode':
            self.pseudo_input = nn.Parameter(
                torch.empty(args.induce_num, *self.tensor_size)
            )
            nn.init.uniform_(self.pseudo_input, 0., 1.)
        else:
            raise NotImplementedError

    def _setup_encoder(self, args):
        encoder = []
        for d in range(self.tensor_dim):
            encoder.append(MLPEncoderDecoder(
                input_size=int(np.prod(self.tensor_size)),
                output_size=[self.tensor_size[d] * args.in_rank ** 2] * 2,
                hidden_size=args.h_dim,
                act=nn.ReLU(),
                out_act=[None, nn.Softplus()],
                bn=False
            ))
        self.encoder = nn.ModuleList(encoder)

    def _setup_decoder(self, args):
        h_decoder = args.h_dim.copy()
        h_decoder.reverse()

        core_len_in = []
        core_len_out = []
        for d, sz in enumerate(self.tensor_size):
            if d != 0:
                core_len_in.append(sz * args.in_rank ** 2)
                core_len_out.append(sz * args.out_rank ** 2)
            else:
                core_len_in.append(args.in_rank ** 2)
                core_len_out.append(args.out_rank ** 2)

        self.decoder = MLPEncoderDecoderList(
            input_size=core_len_in,
            output_size=core_len_out,
            hidden_size=h_decoder,
            act=nn.ELU(),
            out_act=None,
            bn=False
        )

    def encode(self, x):
        if self.conv_pre is not None:
            assert x.ndim == 3
            x = self.conv_pre(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
        guide_list = []
        for d in range(self.tensor_dim):
            mu, sigma = self.encoder[d](torch.flatten(x, start_dim=1))
            mu = mu.view(-1, self.tensor_size[d], self.args.in_rank ** 2)
            sigma = sigma.view(-1, self.tensor_size[d], self.args.in_rank ** 2)
            guide_list.append(Normal(loc=mu, scale=sigma))  # Batch x I_d x R
        return guide_list

    def decode(self, z):
        particles = z[0].shape[0]
        z_ = []
        for d, zi in enumerate(z):
            assert zi.ndim == 4  # Sample x Batch x I x R
            if d != 0:
                z_.append(torch.flatten(zi, start_dim=2))
            else:
                z_.append(torch.flatten(zi, start_dim=3))

        cores = self.decoder(z_)  # Sample x Batch x R
        for d in range(self.tensor_dim):
            cores[d] = cores[d].reshape(
                particles, -1, self.tensor_size[d],
                self.args.out_rank, self.args.out_rank
            )
        mu = full_batch_tr(cores)
        if self.conv_post is not None:
            mu = self.conv_post(mu.view(-1, *self.args.input_shape)
                                ).view(particles, -1, *self.args.input_shape)
        if self.args.input_type in ['binary', 'color']:
            mu = torch.sigmoid(mu)
        mu = mu.view(particles, -1, *self.args.input_shape)
        return mu

    def reparameterization(self, guide_list, num=None):
        if num is None:
            num = self.num_particles
        z_list = []
        for d in range(self.tensor_dim):
            z_list.append(guide_list[d].rsample([num]))
        return z_list  # Sample x Batch x Time x Dim x Rank

    def kl_divergence(self, guide_list):
        if isinstance(self.prior, Distribution):
            kl = 0.
            for guide in guide_list:
                kl = kl + torch.distributions.kl_divergence(guide, self.prior).sum([-2, -1])
        elif isinstance(self.prior, MatrixGP):
            kl = kl_normal_gp(guide_list, self.prior)
        elif isinstance(self.prior, SparseMatrixGP):
            log_q = 0.
            for d in range(self.tensor_dim):
                log_q = log_q + guide_list[d].log_prob(self.z_sample[d]).sum([-2, -1])
            log_p = self.prior.logp
            kl = self.beta * (log_q - log_p)
        else:
            raise NotImplementedError
        return kl

    def sample_latent(self, num):
        z_list = []
        for d in range(self.tensor_dim):
            z_list.append(self.prior.sample([num, self.tensor_size[d]]))
        return z_list
