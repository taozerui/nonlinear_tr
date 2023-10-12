import torch
import math


def vec(t: torch.Tensor):
    return torch.from_numpy(t.numpy().reshape(-1, order='F'))


def gaussian_nll_loss(x, x_mean, x_var, reduction='none', eps=1e-10):
    # Entries of var must be non-negative
    if torch.any(x_var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = x_var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    # Calculate the loss
    loss = 0.5 * (torch.log(var) + (x_mean - x) ** 2 / var)
    loss += 0.5 * math.log(2 * math.pi)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def kl_normal_matrix_normal(p, q_cov1, q_cov2):
    """
    KL divergence between a normal distribution p and multivariate normal distribution q.
    KL(p \lVert q(0, q_cov1, q_cov2))
    :param p: I \times N \times R
    :param q_cov1: I \times I
    :param q_cov2: N \times N
    :return:
    """
    i1 = q_cov1.shape[0]
    i2 = q_cov2.shape[0]
    q_cov1_inv = torch.inverse(q_cov1)
    q_cov2_inv = torch.inverse(q_cov2)
    kl_div = torch.einsum(
        'n, inr, i->r', torch.diag(q_cov2_inv), p.scale ** 2, torch.diag(q_cov1_inv)
    )  # tr((K^{-1} \otimes \Omega^{-1}) diag(vec(p.scale.t()) ** 2))

    kl_div = kl_div + torch.einsum(
        'mn, inr, ij, jmr-> r', q_cov2_inv, p.loc, q_cov1_inv, p.loc
    ) - i1 * i2

    kl_div = kl_div + i2 * torch.logdet(q_cov1) + i1 * torch.logdet(q_cov2)
    kl_div = kl_div - 2 * torch.log(p.scale).sum([0, 1])

    return 0.5 * kl_div


def kl_normal_matrix_normal_logdiag(p, q_cov1_log_diag, q_cov2):
    """
    Same with kl_normal_matrix_normal, but q_cov1 is a diagonal matrix with log elements.
    """
    i1 = len(q_cov1_log_diag)
    i2 = q_cov2.shape[0]
    q_cov1_inv_diag = 1. / torch.exp(q_cov1_log_diag)
    q_cov2_inv = torch.inverse(q_cov2)
    kl_div = torch.einsum(
        'n, inr, i->r', torch.diag(q_cov2_inv), p.scale ** 2, q_cov1_inv_diag
    )  # tr((K^{-1} \otimes \Omega^{-1}) diag(vec(p.scale.t()) ** 2))

    kl_div = kl_div + torch.einsum(
        'mn, inr, i, imr-> r', q_cov2_inv, p.loc, q_cov1_inv_diag, p.loc
    ) - i1 * i2

    kl_div = kl_div + i2 * q_cov1_log_diag.sum() + i1 * torch.logdet(q_cov2)
    kl_div = kl_div - 2 * torch.log(p.scale).sum([0, 1])

    return 0.5 * kl_div


if __name__ == '__main__':
    from torch.distributions import Normal, MultivariateNormal
    import time


    def test_kl_div(batch=10):
        N = 5
        I = 6
        mu_p = torch.randn(I, N, batch)
        sig_p = torch.rand(I, N, batch)
        p = Normal(loc=mu_p, scale=sig_p)

        q_cov1 = torch.randn(I, I)
        q_cov1 = torch.matmul(q_cov1, q_cov1.t())
        q_cov2 = torch.randn(N, N)
        q_cov2 = torch.matmul(q_cov2, q_cov2.t())

        tic = time.time()
        foo = kl_normal_matrix_normal(p, q_cov1, q_cov2)
        toc = time.time()
        print(f'Time cost {toc - tic}.')

        tic = time.time()
        bar = []
        for i in range(batch):
            p = MultivariateNormal(
                loc=vec(mu_p[:, :, i]),
                covariance_matrix=torch.diag(vec(sig_p[:, :, i]) ** 2)
            )
            q = MultivariateNormal(
                loc=torch.zeros(N * I),
                covariance_matrix=torch.kron(q_cov2, q_cov1)
            )
            bar.append(torch.distributions.kl_divergence(p, q))
        bar = torch.tensor(bar)
        toc = time.time()
        print(f'Time cost {toc - tic}.')

        print(foo)
        print(bar)

        print((foo - bar).norm() / foo.norm())

    test_kl_div()
