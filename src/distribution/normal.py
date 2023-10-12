import torch
from torch.distributions import MultivariateNormal, Normal


@torch.distributions.register_kl(MultivariateNormal, Normal)
def _kl_multivariatenormal_normal(p, q):
    kl = torch.logdet(p.covariance_matrix) - torch.log
    pass
