import torch
import math
from torch import nn
from torch.distributions import Normal

from typing import List, Union


class GPPrior(nn.Module):
    """ GPPrior """
    def __init__(self, kernel):
        super(GPPrior, self).__init__()
        self.kernel = kernel

        self.tau = nn.Parameter(torch.tensor([1000.]), requires_grad=False)  # inverse variance (precision) of GP

    def log_prob(self, x, y):
        """
        x: input feature or latent variables.
        y: output.
        """
        raise NotImplementedError

    def forward(self, x, y):
        return self.log_prob(x, y)


class MatrixGP(GPPrior):
    """
    MatrixGP noise free version
    This class implement $D$ separate MatrixGP.
    """
    def __init__(self, kernel, shape: List):
        super(MatrixGP, self).__init__(kernel)
        self.shape = shape  # [I_1, ..., I_D]
        self.D = len(shape)

        self.k_logdet = None  # determinant of kernel matrix
        self.k_inverse = None  # inverse kernel, to compute KL divergence

        # diagonal precision matrix
        precision = []
        for i in shape:
            precision.append(
                nn.Parameter(torch.empty(i), requires_grad=True)
            )
        self.precision = nn.ParameterList(precision)
        self._init_params()

    def _init_params(self):
        for i in self.precision:
            nn.init.normal_(i, mean=1., std=0.1)

    def log_prob(self, x, y):
        """
        x: N \times feature_len, input feature or latent variables.
        y: List of arryas, Sample \times N \times I_d \times R, output.
        """
        assert len(y) == self.D
        k = self.kernel(x)  # kernel matrix, N \times N
        k_inv = torch.inverse(k)
        k_logdet = torch.logdet(k)
        self.k_logdet = k_logdet
        self.k_inverse = k_inv
        out = []
        for d in range(self.D):
            y_i = y[d]
            assert y_i.ndim == 4, "Sample x Data x Size x Rank"
            # if y_i.ndim == 2:
            #     y_i = y_i.unsqueeze(2)
            dim = y_i.shape[1] * y_i.shape[2]
            precision = self.precision[d] ** 2

            const_part = - 0.5 * dim * math.log(math.pi * 2)  # constant part
            det_part = - .5 * (- torch.log(precision).sum() * y_i.shape[2]
                               + k_logdet * y_i.shape[1])  # determinant part

            error_part = - 0.5 * torch.einsum(
                'ij, sjkb, k, sikb-> sb', k_inv, y_i, precision, y_i
            )
            prob = const_part + det_part + error_part
            out.append(prob)

        return torch.vstack(out)  # D \times Batch


def kl_normal_gp(p: List[Normal], q: MatrixGP):
    """
    KL divergence between a Normal distribution (scalar) and MatrixGP, KL(p \lVert q).
    https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence
    :param p: 
    :param q: 
    :return: 
    """
    assert q.k_inverse is not None
    kl = 0.
    for d, p_d in enumerate(p):
        assert p_d.loc.ndim == 3
        dim = p_d.loc.shape[0] * p_d.loc.shape[1]
        q_precision = q.precision[d] ** 2
        q_cov_diag = diag_kron(q.k_inverse, q_precision)

        kl_d = torch.einsum('j, jb-> b', q_cov_diag, p_d.scale.permute(1, 0, 2).reshape(dim, -1))
        kl_d = kl_d + torch.einsum(
            'ij, jmr, m, imr-> r', q.k_inverse, p_d.loc, q_precision, p_d.loc
        ) - dim
        det_q_cov = len(q.precision[d]) * q.k_logdet - q.k_inverse.shape[0] * torch.log(q_precision).sum()
        det_p_cov = torch.log(p_d.scale).sum([0, 1])
        kl_d = kl_d + det_p_cov - det_q_cov
        kl = kl + 0.5 * kl_d

    return kl.sum()


def diag_kron(a, b):
    """
    Compute the diagonal elemnts of kronecker product.
    :param a:
    :param b:
    :return:
    """
    out = []
    for i in range(a.shape[0]):
        out.append(a[i, i] * b)

    return torch.cat(out)
