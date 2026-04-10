from torchmetrics.functional import pearson_corrcoef
import torch
import numpy as np


target_csv = 'predictions.csv'

path = '/abaw5/MTL_abaw5/'
predict_csv = 'result5_test.csv'

target = torch.from_numpy(np.sort(np.loadtxt(path+target_csv, delimiter=',', skiprows=1)))
predict = torch.from_numpy(np.sort(np.loadtxt(path+predict_csv, delimiter=',', skiprows=1)))

pearson_7 = pearson_corrcoef(predict, target)

print('per emotion pcc : ', pearson_7[1:])

print('mean : ', torch.mean(pearson_7[1:]))


