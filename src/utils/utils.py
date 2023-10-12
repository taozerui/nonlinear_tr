import math
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm


def anneal_beta(anneal_function, step, beta_max=1., k=0.0025, max_step=5000):
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - max_step))))
    elif anneal_function == 'linear':
        return min(beta_max, step / max_step)


def evaluate_model(model, data_loader, args):
    model.to(args.device)
    model.eval()

    particles = args.particles
    mb = args.mini_batch

    if particles < mb:
        eval_round = 1
    else:
        assert particles % mb == 0
        eval_round = particles // mb
        particles = mb
    model.num_particles = particles

    assert data_loader.batch_size == 1
    nelbo = []
    nll = []
    recon = []
    kl = []
    with torch.no_grad():
        for i, (x, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
            elbo_round = []
            recon_round = []
            kl_round = []
            for _ in range(eval_round):
                _ = model(x, ave=False)
                elbo_round.append(- model.loss['NELBO'])
                recon_round.append(model.loss['Recon'])
                kl_round.append(model.loss['KL'])

            ll_round = torch.vstack(elbo_round)
            assert ll_round.shape[0] == particles
            ll_round = torch.logsumexp(ll_round, dim=0) - math.log(particles)

            elbo_round = torch.vstack(elbo_round)
            elbo_round = elbo_round.mean(0)

            recon_round = torch.vstack(recon_round).mean(0)
            kl_round = torch.vstack(kl_round).mean(0)

            nelbo.append(- elbo_round.cpu().numpy())
            nll.append(- ll_round.cpu().numpy())
            recon.append(recon_round.cpu().numpy())
            kl.append(kl_round.cpu().numpy())

    nelbo = np.mean(nelbo)
    nll = np.mean(nll)
    recon = np.mean(recon)
    kl = np.mean(kl)

    print('NELBO is {:.3f}, NLL is {:.3f}, Recon is {:.3f}, KL divergence is {:.3f}, Gap is {:.3f}.'.format(
        nelbo, nll, recon, kl, nelbo - nll)
    )
    return nelbo, nll, recon, kl
