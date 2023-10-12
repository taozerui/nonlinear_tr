import torch
import pyro.contrib.gp as gp
from torch import nn
from torch.distributions import Normal
from math import pi

from typing import List

from .utils import kl_normal_matrix_normal, kl_normal_matrix_normal_logdiag

    
def cauchy_kernel(time, sigma, length_scale):
    xs = torch.arange(time)
    xs_in = xs.view(1, -1)
    xs_out = xs.view(-1, 1)
    distance_matrix = (xs_in - xs_out) ** 2
    distance_matrix_scaled = distance_matrix / length_scale ** 2
    kernel_matrix = sigma / (distance_matrix_scaled + 1.)

    alpha = 0.001 * torch.eye(*kernel_matrix.size())

    return kernel_matrix + alpha


class SparseMatrixGP(nn.Module):
    """ MatrixGP """
    def __init__(self, kernel, shape: List, pseudo_num, tau):
        # tau is very important, currently we have
        # tau = 1.0 for MAR
        # tau = 50.0 or 100.0 for MR 0.5
        # tau = 50.0 for MR 0.7
        # tau = 10.0 for physionet
        # tau = 100.0 for air
        super(SparseMatrixGP, self).__init__()
        self.shape = shape
        self.dim = len(shape)
        self.pseudo_num = pseudo_num
        self.feature_len = kernel.input_dim

        self.tau = nn.Parameter(torch.tensor([tau]), requires_grad=False)  # inverse variance (precision) of GP
        self.kernel = kernel

        self.logp = None

        # pseudo feature
        self.pseudo_feature = nn.Parameter(
            torch.empty(pseudo_num, self.feature_len), requires_grad=True
        )

        # omega, spatial information
        # currently, we use diagonal omega.
        # To ensure positive elements, we use log(omega).
        omega = []
        for d in range(self.dim):
            if d == 0:
                omega.append(nn.Parameter(
                    cauchy_kernel(shape[0], 1.005, 3.5), requires_grad=False
                    # torch.empty(shape[d]), requires_grad = True
                ))
            else:
                omega.append(nn.Parameter(
                    torch.empty(shape[d]), requires_grad=True
                ))
                nn.init.normal_(omega[-1], std=0.5)
        self.log_omega = nn.ParameterList(omega)

        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.pseudo_feature, std=1.)
        # for omega in self.log_omega:
        #     nn.init.normal_(omega, std=0.5)

    def log_prob(self, x, y, q_m, q_S):
        """
        x: N \times feature_len, input feature or latent variables.
        y: List of arryas, Sample \times N \times I_d \times R, output.
        q_m: List of arrays, M \times I_d \times R
        """
        knn = self.kernel(x)
        kmm = self.kernel(self.pseudo_feature)  # M x M
        kmm_inv = torch.inverse(kmm)
        kmn = self.kernel(self.pseudo_feature, x)

        sigma_cov = knn - torch.matmul(kmn.t(), torch.matmul(kmm_inv, kmn))
        lambda_cov = torch.einsum(
            'ij, jn, mn, ms-> isn', kmm_inv, kmn, kmn, kmm_inv
        )  # M \times M \times N
        logp = 0.
        omega_trace = 0.
        r = q_m[0].shape[-1]  # rank of the core tensors
        for d in range(self.dim):
            mu = torch.einsum(
                'mir, mj, jn-> inr', q_m[d], kmm_inv, kmn
            )  # I_d \times N \times R^2
            assert y[d].ndim == 4  # Sample x N x I x R
            logp = logp - 0.5 * self.tau * (
                    (mu - y[d].permute([0, 2, 1, 3])) ** 2
            ).sum([-3, -1]) # + 0.5 * (mu.shape[0] * r) * torch.log(self.tau / (2 * pi))  # output shape: Sample
            logp = logp - 0.5 * self.tau * r * torch.einsum(
                'mir, mmn->nir', q_S[d], lambda_cov
            ).sum([-2, -1])
            if self.log_omega[d].ndim == 2:
                omega_trace = omega_trace + torch.trace(self.log_omega[d])
            elif self.log_omega[d].ndim == 1:
                omega_trace = omega_trace + torch.exp(self.log_omega[d]).sum()
            else:
                raise ValueError("Invalid covariance matrix!")

            # KL divergence of inducing points
            if self.log_omega[d].ndim == 2:  # full covariance
                kl_f = kl_normal_matrix_normal(
                    p=Normal(
                        loc=q_m[d].permute([1, 0, 2]).contiguous(),
                        scale=q_S[d].permute([1, 0, 2]).contiguous()
                    ),  # q(\tilde{f})
                    q_cov1=self.log_omega[d], q_cov2=kmm  # p(\tilde{f})
                ).sum()
            elif self.log_omega[d].ndim == 1:  # diagonal covariance
                kl_f = kl_normal_matrix_normal_logdiag(
                    p=Normal(
                        loc=q_m[d].permute([1, 0, 2]).contiguous(),
                        scale=q_S[d].permute([1, 0, 2]).contiguous()
                    ),  # q(\tilde{f})
                    q_cov1_log_diag=self.log_omega[d], q_cov2=kmm  # p(\tilde{f})
                ).sum()
            else:
                raise ValueError("Invalid covariance matrix!")
            logp = logp - kl_f

        logp = logp - 0.5 * self.tau * r * torch.diag(sigma_cov) * omega_trace

        self.logp = logp

        return logp

    def forward(self, x, y, q_m, q_S):
        return self.log_prob(x, y, q_m, q_S)


if __name__ == '__main__':

    import pyro.contrib.gp as gp

    f = 10
    n = 128
    s = 2
    m = 12
    shape = [7, 8, 9]
    r = 4 ** 2
    d = len(shape)

    sparse_gp = SparseMatrixGP(
        kernel=gp.kernels.RBF(input_dim=f),
        shape=shape,
        pseudo_num=m,
    )

    x = torch.randn(n, f)
    y = []
    q_m = []
    q_S = []
    for i in range(d):
        y.append(torch.randn(s, n, shape[i], r))
        q_m.append(torch.randn(m, shape[i], r))
        q_S.append(torch.rand(m, shape[i], r))

    p = sparse_gp(x, y, q_m, q_S)
