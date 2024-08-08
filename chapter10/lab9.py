import numpy as np
import pandas as pd
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

from sklearn.linear_model import (LinearRegression, LogisticRegression, Lasso)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from sklearn.model_selection import (train_test_split, GridSearchCV)
from matplotlib.pyplot import subplots

from utils.summaryPlot import summary_plot

((X_train, Y_train),(X_valid, Y_valid), (X_test, Y_test)) = load_sparse(validation=2000, random_state=0, root='data/IMDB')

lam_max = np.abs(X_train.T * (Y_train - Y_train.mean())).max()
lam_val = lam_max * np.exp(np.linspace(np.log(1),np.log(1e-4), 50))

logit = LogisticRegression(penalty='l1', C=1 / lam_max, solver='liblinear', warm_start=True, fit_intercept=True)

coefs = []
intercepts = []

for l in lam_val:
    logit.C = 1/l
    logit.fit(X_train, Y_train)
    coefs.append(logit.coef_.copy())
    intercepts.append(logit.intercept_)

coefs = np.squeeze(coefs)
intercepts = np.squeeze(intercepts)

fig, axes = subplots(1, 2, figsize=(16, 8), sharey=True)
for ((X_, Y_),
     data_,
     color) in zip([(X_train, Y_train),
                    (X_valid, Y_valid),
                    (X_test, Y_test)],
                    ['Training', 'Validation', 'Test'],
                    ['black', 'red', 'blue']):
    linpred_ = X_ * coefs.T + intercepts[None,:]
    label_ = np.array(linpred_ > 0)
    accuracy_ = np.array([np.mean(Y_ == l) for l in label_.T])
    axes[0].plot(-np.log(lam_val / X_train.shape[0]),
                 accuracy_,
                 '.--',
                 color=color,
                 markersize=13,
                 linewidth=2,
                 label=data_)
axes[0].legend()
axes[0].set_xlabel(r'$-\log(\lambda)$', fontsize=20)
axes[0].set_ylabel('Accuracy', fontsize=20)

imdb_results = pd.read_csv('logs/IMDB/version_0/metrics.csv')
summary_plot(imdb_results,
             axes[1],
             col='accuracy',
             ylabel='Accuracy')
test_accuracy = 0.8492799997329712
axes[1].set_xticks(np.linspace(0, 30, 7).astype(int))
axes[1].set_ylabel('Accuracy', fontsize=20)
axes[1].set_xlabel('Epoch', fontsize=20)
axes[1].set_ylim([0.5, 1]);
axes[1].axhline(test_accuracy,
                color='blue',
                linestyle='--',
                linewidth=3)

plt.show()