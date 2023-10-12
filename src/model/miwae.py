import torch
from torch import nn
from torch.distributions import Normal

from .base import BaseVAE
from .layers import MLPEncoderDecoder


class MIWAE(BaseVAE):
    """ MIWAE """
    def __init__(self, args):
        super(MIWAE, self).__init__(args=args)
        self.__name__ = 'MIWAE'
        self._setup_prior(args)
        self._setup_encoder(args)
        self._setup_decoder(args)

        self._setup_conv_layers(args)

    def _setup_prior(self, args):
        self.prior = Normal(
            loc=torch.zeros(args.z_dim).to(self.device),
            scale=torch.ones(args.z_dim).to(self.device)
        )

    def _setup_encoder(self, args):
        self.encoder = MLPEncoderDecoder(
            input_size=self.input_len,
            output_size=[args.z_dim] * 2,
            hidden_size=args.h_dim,
            act=nn.ReLU(),
            out_act=[None, nn.Softplus()],
            bn=False
        )

    def _setup_decoder(self, args):
        h_decoder = args.h_dim.copy()
        h_decoder.reverse()
        if self.input_type == 'binary':
            out_act = nn.Sigmoid()
        elif self.input_type == 'color':
            out_act = nn.Sigmoid()
        elif self.input_type == 'continuous':
            out_act = None
        else:
            raise NotImplementedError

        self.decoder = MLPEncoderDecoder(
            input_size=args.z_dim,
            output_size=[self.input_len],
            hidden_size=h_decoder,
            act=nn.ELU(),
            out_act=out_act,
            bn=False
        )

    def encode(self, x):
        if self.conv_pre is not None:
            assert x.ndim == 3
            x = self.conv_pre(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)

        # flat data and encode
        x = torch.flatten(x, start_dim=self.flatten)
        mu, sigma = self.encoder(x)
        guide = Normal(mu, sigma)
        return guide

    def decode(self, z):
        particles = z.shape[0]
        mu = self.decoder(z)
        mu = mu.view(particles, -1, *self.args.input_shape)
        return mu

    def reparameterization(self, guide, num=None):
        if num is None:
            num = self.num_particles
        z = guide.rsample([num])
        return z  # Sample x Batch x Dim
