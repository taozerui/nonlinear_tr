import torch
from torch import nn


def unfold(t, start_dim=0, dim=0):
    assert dim >= start_dim
    total_dim = t.ndim

    perm = list(range(start_dim)) + list(range(dim, total_dim)) + list(range(start_dim, dim))
    out = torch.flatten(t.permute(perm), start_dim=start_dim+1)

    return out


def full_tr(cores):
    dim = len(cores)
    sz = []
    for d in range(dim):
        sz.append(cores[d].shape[0])

    out = torch.einsum('imn, jnk-> ijmk', cores[0], cores[1])
    for d in range(2, dim):
        _, _, r1, r2 = out.shape
        out = out.reshape(-1, r1, r2)
        out = torch.einsum('imn, jnk-> ijmk', out, cores[d])

    out = torch.einsum('ijkk-> ij', out)
    out = out.view(sz)
    return out


def sub_full_tr(cores):
    dim = len(cores)
    out = torch.einsum('iab, ibc-> iac', cores[0], cores[1])
    for d in range(2, dim):
        out = torch.einsum('iab, ibc-> iac', out, cores[d])

    out = torch.einsum('iaa-> i', out)
    return out


def full_batch_tr(cores):
    batch = list(cores[0].shape[:-3])
    dim = len(cores)
    sz = []
    for d in range(dim):
        sz.append(cores[d].shape[-3])

    out = torch.einsum('...imn, ...jnk-> ...ijmk', cores[0], cores[1])
    for d in range(2, dim):
        r1, r2 = out.shape[-2:]
        out = out.reshape(*batch, -1, r1, r2)
        out = torch.einsum('...imn, ...jnk-> ...ijmk', out, cores[d])

    out = torch.einsum('...kk-> ...', out)
    out = out.view(batch + sz)
    return out


def near_eye_init(shape, rank, noise=1e-3, learn=True):
    # Initialize core using size-adjusted value of noise
    cores = []
    for i in range(len(shape)):
        eye_core = torch.stack([torch.eye(rank, rank) for _ in range(shape[i])])
        noise = noise / rank
        eye_core += noise * torch.randn(shape[i], rank, rank)
        if learn:
            cores.append(nn.Parameter(eye_core))
        else:
            cores.append(eye_core)

    return cores


if __name__ == '__main__':
    # batch = [12, 24]
    # sz = [3, 4, 5]
    # r = 6
    # cores = []
    # for i in sz:
    #     cores.append(torch.randn(*batch, i, r, r))
    #
    # t = full_batch_tr(cores)
    # foo = torch.einsum(
    #     '...ab, ...bc, ...ca-> ...',
    #     cores[0][:, :, 1, :, :], cores[1][:, :, 2, :, :], cores[2][:, :, 3, :, :]
    # )
    # bar = t[:, :, 1, 2, 3]
    # print((foo - bar).norm() / foo.norm())

    x = torch.randn(128, 5, 24, 28)
    for d in range(x.ndim):
        print(unfold(x, start_dim=0, dim=d).shape)
    for d in range(1, x.ndim):
        print(unfold(x, start_dim=1, dim=d).shape)
