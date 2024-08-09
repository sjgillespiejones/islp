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

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_size, 32)
        self.lstm = nn.LSTM(input_size=32,
                            hidden_size=32,
                            batch_first=True)
        self.dense = nn.Linear(32, 1)
    def forward(self, x):
        val, (h_n, c_n) = self.lstm(self.embedding(x))
        return torch.flatten(self.dense(val[:,-1]))

def main():
    seed_everything(0, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    max_num_workers = rec_num_workers()

    (imdb_seq_train, imdb_seq_test) = load_sequential(root='data/IMDB')
    imdb_seq_dm = SimpleDataModule(imdb_seq_train,imdb_seq_test,validation=2000,batch_size=300,num_workers=min(6, max_num_workers))

    ((X_train, Y_train),
     (X_valid, Y_valid),
     (X_test, Y_test)) = load_sparse(validation=2000,
                                     random_state=0,
                                     root='data/IMDB')

    lstm_model = LSTMModel(X_test.shape[-1])
    summary(lstm_model,input_data=imdb_seq_train.tensors[0][:10],col_names=['input_size','output_size','num_params'])

    lstm_module = SimpleModule.binary_classification(lstm_model)
    lstm_logger = CSVLogger('logs', name='IMDB_LSTM')

    lstm_trainer = Trainer(deterministic=True,
                           max_epochs=20,
                           logger=lstm_logger,
                           callbacks=[ErrorTracker()])
    lstm_trainer.fit(lstm_module,
                     datamodule=imdb_seq_dm)

    lstm_trainer.test(lstm_module, datamodule=imdb_seq_dm)

if __name__ == '__main__':
    main()