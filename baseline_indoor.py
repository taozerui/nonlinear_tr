import pickle
import numpy as np
from fancyimpute import KNN, SoftImpute


def knn_impute():
    with open('./data/process/indoor/indoor.pkl', 'rb') as f:
        data = pickle.load(f)
    x_obs = data['data_train']
    x_obs[data['mask_train'] == 0.] = np.nan
    x_obs = x_obs.reshape(-1, 9 * 2)
    x_hat = KNN(k=3).fit_transform(x_obs).reshape(-1, 9, 2)
    x_hat = x_hat[data['mask_test'] == 1.]
    x_val = data['data_test'][data['mask_test'] == 1.]
    rmse = np.sqrt(np.mean((x_hat - x_val) ** 2))
    mae = np.mean(np.abs(x_hat - x_val))
    return rmse, mae


def soft_impute():
    with open('./data/process/indoor/indoor.pkl', 'rb') as f:
        data = pickle.load(f)
    x_obs = data['data_train']
    x_obs[data['mask_train'] == 0.] = np.nan
    x_obs = x_obs.reshape(-1, 9 * 2)
    x_hat = SoftImpute().fit_transform(x_obs).reshape(-1, 9, 2)
    x_hat = x_hat[data['mask_test'] == 1.]
    x_val = data['data_test'][data['mask_test'] == 1.]
    rmse = np.sqrt(np.mean((x_hat - x_val) ** 2))
    mae = np.mean(np.abs(x_hat - x_val))
    return rmse, mae

soft_result = dict(rmse=[], mae=[])
for i in range(5):
    rmse, mae = soft_impute()
    soft_result['rmse'].append(rmse)
    soft_result['mae'].append(mae)

# with open(f'./baseline_result/indoor/softimpute.pkl', 'wb') as f:
#     pickle.dump(soft_result, f)

knn_result = dict(rmse=[], mae=[])
for i in range(5):
    rmse, mae = knn_impute()
    knn_result['rmse'].append(rmse)
    knn_result['mae'].append(mae)


print(f"SoftImpute: RMSE is {np.mean(soft_result['rmse']):.3f}, MAE is {np.mean(soft_result['mae']):.3f}")
print(f"KNN: RMSE is {np.mean(knn_result['rmse']):.3f}, MAE is {np.mean(knn_result['mae']):.3f}")

# with open(f'./baseline_result/indoor/knn.pkl', 'wb') as f:
#     pickle.dump(knn_result, f)