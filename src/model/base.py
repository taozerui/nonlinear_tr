import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli, Normal

from src.distribution import gaussian_nll_loss, MatrixGP, SparseMatrixGP
from .nn import PermLinear
# from src.distributions import log_logistic_256


class BaseVAE(nn.Module):
    """ BaseVAE """
    def __init__(self, args):
        super(BaseVAE, self).__init__()
        self.args = args
        self.device = torch.device(args.device)
        self.num_particles = args.num_particles
        self.input_time = args.input_time
        self.flatten = args.flatten
        self.input_len = np.prod(args.input_shape[args.flatten-1:])
        self.input_type = args.input_type
        if args.anneal_beta:
            beta = 0.0
        else:
            beta = args.beta_max
        self.beta = beta
        self.input_type = args.input_type

        self.prior = None
        self.posterior = None

        self.loss = None

        if args.input_type == 'continuous':
            self.var_x = nn.Parameter(torch.tensor([1e-3]), requires_grad=False)
        elif args.input_type == 'binary':
            self.var_x = None
        else:
            raise NotImplementedError

    def _setup_prior(self, args):
        raise NotImplementedError

    def _setup_encoder(self, args):
        raise NotImplementedError

    def _setup_decoder(self, args):
        raise NotImplementedError

    def _setup_conv_layers(self, args):
        if args.conv_layer:
            assert len(args.input_shape) == 2, "Only support multivariate times series (Dim=2) now!"
            conv_pre = []
            for i in range(len(args.conv_filter)):
                if i == 0:
                    filter_in = args.input_shape[1]
                else:
                    filter_in = args.conv_filter[i-1]
                conv_pre.append(
                    nn.Conv1d(filter_in, args.conv_filter[i], 3, stride=1, padding='same')
                )
                conv_pre.append(nn.ELU())
            conv_pre.append(PermLinear(args.conv_filter[-1], args.input_shape[1]))
            self.conv_pre = nn.Sequential(*conv_pre)

            # self.conv_pre = nn.Sequential(
            #     nn.Conv1d(args.input_shape[1], 128, 3, stride=1, padding='same'), nn.ELU(),
            #     nn.Conv1d(128, 256, 3, stride=1, padding='same'), nn.ELU(),
            #     nn.Conv1d(256, 256, 3, stride=1, padding='same'),  nn.ELU(),
            #     PermLinear(256, args.input_shape[1])
            # )
            self.conv_post = None
        else:
            self.conv_pre = None
            self.conv_post = None

    def encode(self, x):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError

    def iwae_ave(self, x):
        assert x.shape[0] == self.num_particles
        return torch.logsumexp(x, dim=0) - math.log(self.num_particles)

    def reparameterization(self, guide, num):
        raise NotImplementedError

    def impute(self, x, num=None):
        if num is None:
            num = self.num_particles
        guide = self.encode(x)
        z_sample = self.reparameterization(guide, num)
        # log_p = self.prior(z_sample)
        x_mean = self.decode(z_sample)

        return x_mean

    def forward(self, x, latent_vec=None, mask=None, ave=True):
        guide = self.encode(x)
        z_sample = self.reparameterization(guide)
        self.z_sample = z_sample

        if isinstance(self.prior, MatrixGP):
            self.prior(latent_vec, z_sample)
        elif isinstance(self.prior, SparseMatrixGP):
            if self.args.induce_type == 'simple':
                q_S = [torch.exp(self.q_S[d]) for d in range(self.tensor_dim)]
                self.prior(latent_vec, z_sample, self.q_m, q_S)
            else:
                q_m = []
                q_S = []
                cache = self.encode(self.pseudo_input)
                for d in range(self.tensor_dim):
                    q_m.append(cache[d].loc)
                    q_S.append(cache[d].scale + 1. / self.prior.tau)
                self.prior(latent_vec, z_sample, q_m, q_S)
        else:
            pass

        x_mean = self.decode(z_sample)
        if self.input_type == 'binary':
            self.posterior = Bernoulli(probs=x_mean.mean(0))
        else:
            self.posterior = Normal(loc=x_mean.mean(0), scale=self.var_x)

        if self.training:
            beta = self.beta
        else:
            beta = 1.0

        # flat data
        x_flat = torch.flatten(x, start_dim=1)
        x_mean_flat = torch.flatten(x_mean, start_dim=2)
        if mask is not None:
            mask = torch.flatten(mask, start_dim=1)
        elbo, ll, kl = self.elbo(x_flat, x_hat=x_mean_flat, mask=mask,
                                 guide=guide,
                                 beta=beta, ave=ave)
        self.loss = {
            'NELBO': - elbo, 'Recon': - ll, 'KL': kl
        }
        return x_mean  # Sample x Batch x Dim

    def elbo(self, x, x_hat, mask, guide, beta=1.0, ave=True):
        assert x_hat.ndim == 3, "Shape should be (Sample, Batch, Dim)"

        # compute log likelihood function
        if self.input_type == 'binary':
            ll = - F.binary_cross_entropy(
                x_hat, x.unsqueeze(0).repeat(self.num_particles, 1, 1), reduction='none'
            )
        elif self.input_type == 'continuous':
            ll = - gaussian_nll_loss(
                x=x.unsqueeze(0).repeat(self.num_particles, 1, 1), x_mean=x_hat, x_var=self.var_x,
                reduction='none'
            )
        else:
            raise NotImplementedError

        # masking
        if mask is not None:
            ll = (ll * mask).sum(-1)
        else:
            ll = ll.sum(-1)

        assert ll.shape == (self.num_particles, x.shape[0])

        kl = self.kl_divergence(guide)
        elbo = ll - kl

        if ave:
            return self.iwae_ave(elbo).mean(), ll.mean(), kl.mean()
        else:
            return elbo, ll, kl

    def kl_divergence(self, guide):
        kl = torch.distributions.kl.kl_divergence(guide, self.prior)
        kl = torch.flatten(kl, start_dim=1).sum([-1])
        return kl

    def sample_latent(self, num):
        z = self.prior.sample([num])
        return z

    def sample(self, num):
        z = self.sample_latent(num)
        x_mean = self.decode(z)

        return x_mean