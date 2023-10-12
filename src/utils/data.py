import pickle
import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader

# np.random.seed(123)
# torch.manual_seed(42)


class MyTensorDataset(TensorDataset):
    """ MyTensorDataset """
    def __init__(self, *tensors):
        super(MyTensorDataset, self).__init__(*tensors)

    def __getitem__(self, index):
        return tuple([index]) + tuple(tensor[index] for tensor in self.tensors)


def construct_time_series(x, len=36):
    t = x.shape[0]
    out = []
    for i in range(t - len):
        out.append(x[i:i+len])

    return np.stack(out)


def get_dataloader(args):
    root_path = args.data_path

    if args.dataset == 'radar':
        with open(os.path.join(root_path, f'radar.pkl'), 'rb') as f:
            dt = pickle.load(f)
        x_train = torch.tensor(construct_time_series(dt['data_train'], 25), dtype=torch.float32).flatten(2)
        x_val = torch.tensor(construct_time_series(dt['data_val'], 25), dtype=torch.float32).flatten(2)
        x_test = torch.tensor(construct_time_series(dt['data_test'], 25), dtype=torch.float32).flatten(2)
        mask_train = torch.tensor(construct_time_series(dt['mask_train'], 25), dtype=torch.float32).flatten(2)
        mask_val = torch.tensor(construct_time_series(dt['mask_val'], 25), dtype=torch.float32).flatten(2)
        mask_test = torch.tensor(construct_time_series(dt['mask_test'], 25), dtype=torch.float32).flatten(2)

        train = MyTensorDataset(x_train, mask_train)
        validation = MyTensorDataset(x_train, x_val, mask_val)
        test = MyTensorDataset(x_train, x_test, mask_test)
    elif args.dataset == 'indoor':
        with open(os.path.join(root_path, f'indoor.pkl'), 'rb') as f:
            dt = pickle.load(f)
        x_train = torch.tensor(construct_time_series(dt['data_train'], 25), dtype=torch.float32).flatten(2)
        x_val = torch.tensor(construct_time_series(dt['data_val'], 25), dtype=torch.float32).flatten(2)
        x_test = torch.tensor(construct_time_series(dt['data_test'], 25), dtype=torch.float32).flatten(2)
        mask_train = torch.tensor(construct_time_series(dt['mask_train'], 25), dtype=torch.float32).flatten(2)
        mask_val = torch.tensor(construct_time_series(dt['mask_val'], 25), dtype=torch.float32).flatten(2)
        mask_test = torch.tensor(construct_time_series(dt['mask_test'], 25), dtype=torch.float32).flatten(2)

        train = MyTensorDataset(x_train, mask_train)
        validation = MyTensorDataset(x_train, x_val, mask_val)
        test = MyTensorDataset(x_train, x_test, mask_test)
    else:
        raise ValueError

    batch_size = args.batch_size

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
    if test is not None:
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    else:
        test_loader = None

    data_loaders = {
        "train": train_loader,
        "test": test_loader,
        "validate": valid_loader
    }
    return data_loaders
