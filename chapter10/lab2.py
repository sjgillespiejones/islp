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

from utils.summaryPlot import summary_plot

seed_everything(0, workers=True)
torch.use_deterministic_algorithms(True, warn_only=True)

from torchvision.datasets import MNIST, CIFAR100
from torchvision.models import (resnet50,ResNet50_Weights)
from torchvision.transforms import (Resize,Normalize , CenterCrop , ToTensor)

from ISLP.torch import (SimpleDataModule , SimpleModule ,ErrorTracker , rec_num_workers)
from ISLP.torch.imdb import (load_lookup ,load_tensor , load_sparse ,load_sequential)

from glob import glob
import json
import matplotlib.pyplot as plt

hit_logger = CSVLogger('logs', name='hitters')
#print(hit_logger.experiment.metrics_file_path)
hit_results = pd.read_csv('logs/hitters/version_4/metrics.csv')

fix, ax = subplots(1, 1, figsize=(6,6))
ax = summary_plot(hit_results, ax, col='mae', ylabel='MAE', valid_legend='Validation (=Test)')
ax.set_ylim([0, 400])
ax.set_xticks(np.linspace(0, 50, 11).astype(int))
plt.show()