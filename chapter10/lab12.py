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

from torchvision.datasets import MNIST, CIFAR100
from torchvision.models import (resnet50,ResNet50_Weights)
from torchvision.transforms import (Resize,Normalize , CenterCrop , ToTensor)

from ISLP.torch import (SimpleDataModule , SimpleModule ,ErrorTracker , rec_num_workers)
from ISLP.torch.imdb import (load_lookup ,load_tensor , load_sparse ,load_sequential)

from glob import glob
import json
import matplotlib.pyplot as plt

class NYSEModel(nn.Module):
    def __init__(self):
        super(NYSEModel, self).__init__()
        self.rnn = nn.RNN(3,
                          12,
                          batch_first=True)
        self.dense = nn.Linear(12, 1)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        val, h_n = self.rnn(x)
        val = self.dense(self.dropout(val[:,-1]))
        return torch.flatten(val)
nyse_model = NYSEModel()

class NonLinearARModel(nn.Module):
    def __init__(self):
        super(NonLinearARModel, self).__init__()
        self._forward = nn.Sequential(nn.Flatten(),
                                      nn.Linear(20, 32),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(32, 1))
    def forward(self, x):
        return torch.flatten(self._forward(x))

def main():
    seed_everything(0, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    max_num_workers = rec_num_workers()

    NYSE = load_data('NYSE')
    cols = ['DJ_return', 'log_volume', 'log_volatility']
    X = pd.DataFrame(StandardScaler(with_mean=True, with_std=True).fit_transform(NYSE[cols]),
                     columns=NYSE[cols].columns, index=NYSE.index)

    for lag in range(1, 6):
        for col in cols:
            newcol = np.zeros(X.shape[0]) * np.nan
            newcol[lag:] = X[col].values[:-lag]
            X.insert(len(X.columns), "{0}_{1}".format(col, lag), newcol)
    X.insert(len(X.columns), 'train', NYSE['train'])
    X = X.dropna()

    Y, train = X['log_volume'], X['train']
    X = X.drop(columns=['train'] + cols)

    M = LinearRegression()
    M.fit(X[train], Y[train])
    # print(M.score(X[~train], Y[~train]))

    X_day = pd.merge(X, pd.get_dummies(NYSE['day_of_week']), on='date')

    M.fit(X_day[train], Y[train])
    # print(M.score(X_day[~train], Y[~train]))

    ordered_cols = []
    for lag in range(5, 0, -1):
        for col in cols:
            ordered_cols.append('{0}_{1}'.format(col, lag))
    X = X.reindex(columns=ordered_cols)
    # print(X.columns)

    X_rnn = X.to_numpy().reshape((-1, 5, 3))
    # print(X_rnn.shape)

    datasets = []
    for mask in [train, ~train]:
        X_rnn_t = torch.tensor(X_rnn[mask].astype(np.float32))
        Y_t = torch.tensor(Y[mask].astype(np.float32))
        datasets.append(TensorDataset(X_rnn_t, Y_t))
    nyse_train, nyse_test = datasets

    summary(nyse_model,input_data=X_rnn_t,col_names=['input_size','output_size','num_params'])

    nyse_dm = SimpleDataModule(nyse_train,nyse_test,num_workers=min(4, max_num_workers),validation=nyse_test,batch_size=64)

    for idx, (x, y) in enumerate(nyse_dm.train_dataloader()):
        out = nyse_model(x)
        print(y.size(), out.size())
        if idx >= 2:
            break

    # nyse_optimizer = RMSprop(nyse_model.parameters(),lr=0.001)
    # nyse_module = SimpleModule.regression(nyse_model,optimizer=nyse_optimizer,metrics={'r2': R2Score()})
    #
    # nyse_trainer = Trainer(accelerator='mps',deterministic=True,max_epochs=200,callbacks=[ErrorTracker()])
    # nyse_trainer.fit(nyse_module,datamodule=nyse_dm)
    # nyse_trainer.test(nyse_module,datamodule=nyse_dm)

    datasets = []
    for mask in [train, ~train]:
        X_day_t = torch.tensor(
            np.asarray(X_day[mask]).astype(np.float32))
        Y_t = torch.tensor(np.asarray(Y[mask]).astype(np.float32))
        datasets.append(TensorDataset(X_day_t, Y_t))
    day_train, day_test = datasets

    day_dm = SimpleDataModule(day_train,day_test,num_workers=min(4, max_num_workers),validation=day_test,batch_size=64)
    nl_model = NonLinearARModel()
    nl_optimizer = RMSprop(nl_model.parameters(),lr=0.001)
    nl_module = SimpleModule.regression(nl_model,optimizer=nl_optimizer,metrics={'r2': R2Score()})
    nl_trainer = Trainer(accelerator='mps',deterministic=True,max_epochs=20,callbacks=[ErrorTracker()])
    nl_trainer.fit(nl_module, datamodule=day_dm)
    nl_trainer.test(nl_module, datamodule=day_dm)


if __name__ == '__main__':
    main()