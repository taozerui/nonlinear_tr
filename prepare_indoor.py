import pickle
import numpy as np
import pandas as pd
from scipy.io import savemat

np.random.seed(42)

train_val = np.load('./data/indoor/train_vals.npy')
train_idx = np.load('./data/indoor/train_idxs.npy')
test_val = np.load('./data/indoor/test_vals.npy')
test_idx = np.load('./data/indoor/test_idxs.npy')
val_val = np.load('./data/indoor/valid_vals.npy')
val_idx = np.load('./data/indoor/valid_idxs.npy')

x_train = np.zeros([19735, 9, 2])
mask_train = np.zeros_like(x_train)
for i in range(train_val.shape[0]):
    x_train[train_idx[i, 0], train_idx[i, 1], train_idx[i, 2]] = train_val[i]
    mask_train[train_idx[i, 0], train_idx[i, 1], train_idx[i, 2]] = 1.

x_test = np.zeros([19735, 9, 2])
mask_test = np.zeros_like(x_test)
for i in range(test_val.shape[0]):
    x_test[test_idx[i, 0], test_idx[i, 1], test_idx[i, 2]] = test_val[i]
    mask_test[test_idx[i, 0], test_idx[i, 1], test_idx[i, 2]] = 1.

x_val = np.zeros([19735, 9, 2])
mask_val = np.zeros_like(x_val)
for i in range(val_val.shape[0]):
    x_val[val_idx[i, 0], val_idx[i, 1], val_idx[i, 2]] = val_val[i]
    mask_val[val_idx[i, 0], val_idx[i, 1], val_idx[i, 2]] = 1.

data_store = {
    'data_train': x_train, 'mask_train': mask_train,
    'data_val': x_val, 'mask_val': mask_val,
    'data_test': x_test, 'mask_test': mask_test,
}

with open(f'./data/process/indoor/indoor.pkl', 'wb') as f:
    pickle.dump(data_store, f)
# savemat(f'./data/process/indoor/indoor.mat', data_store)
