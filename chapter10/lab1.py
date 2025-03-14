import numpy as np, pandas as pd
from matplotlib.pyplot import subplots
from sklearn.linear_model import (LinearRegression, LogisticRegression, Lasso)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from sklearn.model_selection import (train_test_split, GridSearchCV)

import torch
from torch import nn
from torch.optim import RMSprop
from torch.utils.data import TensorDataset

from torchmetrics import (MeanAbsoluteError , R2Score)
from torchinfo import summary
from torchvision.io import read_image

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from pytorch_lightning import seed_everything
seed_everything(0, workers=True)
torch.use_deterministic_algorithms(True, warn_only=True)

from torchvision.datasets import MNIST, CIFAR100
from torchvision.models import (resnet50,ResNet50_Weights)
from torchvision.transforms import (Resize,Normalize , CenterCrop , ToTensor)

from ISLP.torch import (SimpleDataModule , SimpleModule ,ErrorTracker , rec_num_workers)
from ISLP.torch.imdb import (load_lookup ,load_tensor , load_sparse ,load_sequential)

from glob import glob
import json

Hitters = load_data('Hitters').dropna()
n = Hitters.shape[0]

model = MS(Hitters.columns.drop('Salary'), intercept=False)
X = model.fit_transform(Hitters).to_numpy()
Y = Hitters['Salary'].to_numpy()

(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=1/3, random_state=1)

hit_lm = LinearRegression().fit(X_train, Y_train)
Yhat_test = hit_lm.predict(X_test)
mean_error = np.abs(Yhat_test - Y_test).mean()
print(mean_error)

scaler = StandardScaler(with_mean=True, with_std=True)
lasso = Lasso(warm_start=True, max_iter=30_000)
standard_lasso = Pipeline(steps=[('scaler', scaler), ('lasso', lasso)])

X_s = scaler.fit_transform(X_train)
n = X_s.shape[0]
lam_max = np.fabs(X_s.T.dot(Y_train - Y_train.mean())).max() / n
params_grid = { 'alpha' : np.exp(np.linspace(0, np.log(0.01), 100)) * lam_max}

cv = KFold(10, shuffle=True, random_state=1)
grid = GridSearchCV(lasso, params_grid, cv=cv, scoring='neg_mean_absolute_error')
grid.fit(X_train, Y_train)
# print(grid.best_estimator_)

trained_lasso = grid.best_estimator_
Yhat_test = trained_lasso.predict(X_test)
mean_error = np.fabs(Yhat_test - Y_test).mean()
print(mean_error)


