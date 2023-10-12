import torch
import pyro.contrib.gp as gp
from torch.distributions import MultivariateNormal

from .miwae import MIWAE


class GPVAE(MIWAE):
    """ GPVAE """
    def __init__(self, args):
        super(GPVAE, self).__init__(args)
        self.__name__ = 'GPVAE'

    def _setup_prior(self, args):
        kernel = gp.kernels.RBF(input_dim=self.input_time)
        sigma = kernel(torch.arange(0, self.input_time)).to(self.device)
        self.prior = MultivariateNormal(
            loc=torch.zeros(self.input_time).to(self.device),
            covariance_matrix=sigma.detach()
        )

    def kl_divergence(self, guide):
        dim = self.input_time
        kl = torch.logdet(self.prior.covariance_matrix) - torch.log(guide.scale).sum([-2]) - dim
        inv_prior_cov = torch.linalg.inv(self.prior.covariance_matrix)
        kl = kl + torch.einsum('ii, bid-> bd', inv_prior_cov, guide.scale)
        kl = kl + torch.einsum('bid, ij, bjd-> bd',  # TODO: Add non-zero prior loc
                               guide.loc, inv_prior_cov, guide.loc)
        kl = 0.5 * kl.sum(-1)
        return kl

    def sample_latent(self, num):
        z = self.prior.sample([num, self.args.z_dim])
        z = z.permute(0, 2, 1)
        return z
