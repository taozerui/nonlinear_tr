import numpy as np
import torch

from torch import nn
from tqdm import tqdm

from .utils import anneal_beta


class RealTrainer:
    """ RealTrainer """
    def __init__(self, args):
        super(RealTrainer, self).__init__()
        self.args = args

        self.epoch_count = 1
        self.iter_count = 1
        self.reach_maxiter = False
        # latent vector, for GP prior
        self.latent_vec = None

        self.rmse = {'val': [], 'test': []}
        self.mae = {'val': [], 'test': []}

    def _init_latent_vec(self, num, feature_len, device='cpu'):
        self.latent_vec = nn.Parameter(
            torch.empty(num, feature_len, device=torch.device(device)), requires_grad=True)
        nn.init.normal_(self.latent_vec, std=1.0)

    def train(self, model, train_data, val_data=None, test_data=None):
        args = self.args
        if args.prior == 'sparse_gp':
            self._init_latent_vec(num=train_data.dataset.tensors[0].shape[0],
                                  feature_len=args.latent_feature, device=args.device)
            optimizer = torch.optim.Adam([*model.parameters()] + [self.latent_vec], lr=args.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        while not self.reach_maxiter:
            # train
            model.train()
            bar = tqdm(train_data, desc=f'{model.__name__} Epoch {self.epoch_count} train'.ljust(20))
            for i, (batch_idx, x, mask) in enumerate(bar):
                if args.device != 'cpu':
                    x = x.to(torch.device(args.device))
                    mask = mask.to(torch.device(args.device))
                if self.latent_vec is not None:
                    model(x=x, latent_vec=self.latent_vec[batch_idx], mask=mask)
                else:
                    model(x, mask=mask)

                optimizer.zero_grad()
                model.loss['NELBO'].backward()
                optimizer.step()

                # iteration
                if self.iter_count >= args.max_iter:
                    print('Reach max iteration {}.'.format(args.max_iter))
                    self.reach_maxiter = True
                    break

                # anneal learning rate
                if args.anneal_lr:
                    if self.iter_count == int(args.max_iter * 0.6):
                        optimizer.param_groups[0]['lr'] *= 0.3
                    if self.iter_count == int(args.max_iter * 0.75):
                        optimizer.param_groups[0]['lr'] *= 0.3
                    if self.iter_count == int(args.max_iter * 0.9):
                        optimizer.param_groups[0]['lr'] *= 0.3

                # anneal beta
                if args.anneal_beta:
                    model.beta = anneal_beta(
                        anneal_function='linear',
                        step=self.iter_count,
                        beta_max=args.beta_max,
                        max_step=int(0.3 * args.max_iter)
                    )

                if i % 50 == 0:
                    bar.set_postfix(loss='{:.2f}'.format(model.loss['NELBO'].item()))

                self.iter_count += 1

            model.eval()

            # validate
            if val_data is not None:
                self.validate(model, val_data)
            if test_data is not None:
                self.validate(model, test_data, phase='test')

            self.epoch_count += 1

    @torch.no_grad()
    def validate(self, model, val_data, phase='val'):
        model.eval()

        mse = 0.
        mae = 0.
        obs_num = 0
        for i, (batch_idx, x, x_full, mask) in enumerate(val_data):
            if self.args.device != 'cpu':
                x = x.to(torch.device(self.args.device))
                x_full = x_full.to(torch.device(self.args.device))
                mask = mask.to(torch.device(self.args.device))
            x_hat = model.impute(x, num=1)

            x_hat = x_hat.mean(0)
            mse += (((x_full - x_hat) * mask) ** 2).sum()
            mae += torch.abs((x_full - x_hat) * mask).sum()
            obs_num += mask.sum()

        mse /= obs_num
        mae /= obs_num
        self.rmse[phase].append(np.sqrt(mse.item()))
        self.mae[phase].append(mae.item())

        print('{}: Epoch {} -  RMSE {:.3f} | MAE {:.3f}.'.format(
              phase, self.epoch_count, np.sqrt(mse.item()), mae))
