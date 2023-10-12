import os
import pickle
import numpy as np
import torch
import datetime

torch.manual_seed(1234)
np.random.seed(42)


def main(args):
    # config dataset
    if args.dataset in ['radar']:
        if args.model_type == 'gpvae':
            args.flatten = 2
        else:
            args.flatten = 1
        args.input_time = 25
        args.input_shape = [25, 23 * 5]  # (T, C, W, H)
        args.input_type = 'continuous'
        if args.tensorize:
            args.tensor_size = [25, 23, 5]
        else:
            args.tensor_size = [25, 23 * 5]
    elif args.dataset in ['indoor']:
        if args.model_type == 'gpvae':
            args.flatten = 2
        else:
            args.flatten = 1
        args.input_time = 25
        args.input_shape = [25, 9 * 2]  # (T, C, W, H)
        args.input_type = 'continuous'
        if args.tensorize:
            args.tensor_size = [25, 9, 2]
        else:
            args.tensor_size = [25, 9 * 2]
    else:
        raise KeyError
    args.data_path = os.path.join(args.data_path, args.dataset)

    from src.utils import get_dataloader

    if args.model_type == 'miwae':
        from src.model import MIWAE as VAE
    elif args.model_type == 'gptr':
        from src.model import VariationalTR as VAE
    elif args.model_type == 'gpvae':
        from src.model import GPVAE as VAE
    elif args.model_type == 'tr':
        from src.model import TR as VAE
    else:
        raise NotImplementedError

    data_loader = get_dataloader(args)
    model = VAE(args)
    if args.device != 'cpu':
        model = model.to(torch.device(args.device))

    from src.utils import RealTrainer as Trainer

    trainer = Trainer(args)
    trainer.train(model, data_loader['train'], data_loader['validate'], data_loader['test'])
    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, default='indoor',
                        choices=['indoor', 'radar'])
    parser.add_argument('--data_path', type=str, default='./data/process/')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda',
                        help='Set the device (cpu or cuda, default: cpu)')

    # model
    parser.add_argument('--model_type', type=str, default='miwae',
                        help='miwae, gptr')
    parser.add_argument('--num_particles', type=int, default=1,
                        help='Sample numbers for IWAE bound')
    parser.add_argument('--z_dim', type=int, default=100,
                        help='Latent variable dimension.')
    parser.add_argument('--h_dim', type=int, nargs='+', default=[300, 300],
                        help='Hidden layer size')
    parser.add_argument('--tensorize', default=False, action='store_true')
    parser.add_argument('--in_rank', type=int, default=5)
    parser.add_argument('--out_rank', type=int, default=10)
    parser.add_argument('--conv_layer', default=False, action='store_true')
    parser.add_argument('--conv_filter', type=int, nargs='+', default=[128, 128],
                        help='Hidden layer size')

    # prior
    parser.add_argument('--prior', type=str, default='sparse_gp',
                        help='normal, simple_gp, sparse_gp.')
    parser.add_argument('--latent_feature', type=int, default=5,
                        help='Latent feature length for GP prior.')
    parser.add_argument('--induce_num', type=int, default=30)
    parser.add_argument('--induce_type', type=str, default='simple',
                        help='simple, encode')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='precision (inverse variance) of the core tensors.')

    # train
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_iter', type=int, default=40000)
    parser.add_argument("--anneal_lr", default=False, action='store_true',
                        help='Anneal learning rate')
    parser.add_argument("--anneal_beta", default=False, action='store_true',
                        help='Anneal beta, the scale of KL divergence')
    parser.add_argument("--beta_max", type=float, default=1.)

    args = parser.parse_args()

    model = main(args)
