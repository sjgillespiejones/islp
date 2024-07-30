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

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Dropout(0.4))
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3))
        self._forward = nn.Sequential(
            self.layer1,
            self.layer2,
            nn.Linear(128, 10))
    def forward(self, x):
        return self._forward(x)

class MNIST_MLR(nn.Module):
    def __init__(self):
        super(MNIST_MLR, self).__init__()
        self.linear = nn.Sequential(nn.Flatten(),
                                    nn.Linear(784, 10))
    def forward(self, x):
        return self.linear(x)

def main():
    seed_everything(0, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    max_num_workers = rec_num_workers()

    (mnist_train, mnist_test) = [MNIST(root='data', train=train, download=True, transform=ToTensor()) for train in
                                 [True, False]]

    mnist_dm = SimpleDataModule(mnist_train, mnist_test, validation=0.2, num_workers=max_num_workers, batch_size=256)

    # for idx, (X_, Y_) in enumerate(mnist_dm.train_dataloader()):
    #     print('X: ', X_.shape)
    #     print('Y: ', Y_.shape)
    #     if idx >= 1:
    #         break

    # print(X_)
    mnist_model = MNISTModel()
    # print(mnist_model(X_).size())

    # print(summary(mnist_model, input_data=X_, col_names=['input_size', 'output_size', 'num_params']))

    # mnist_module = SimpleModule.classification(mnist_model, num_classes=10)
    # mnist_logger = CSVLogger('logs', name='MNIST')
    #
    # mnist_trainer = Trainer(deterministic=True, max_epochs=30, logger=mnist_logger, callbacks=[ErrorTracker()])
    # mnist_trainer.fit(mnist_module, datamodule=mnist_dm)
    #
    # print(mnist_trainer.test(mnist_module, datamodule=mnist_dm))

    mlr_model = MNIST_MLR()
    mlr_module = SimpleModule.classification(mlr_model, num_classes=10)

    mlr_logger = CSVLogger('logs', name='MNIST_MLR')

    mlr_trainer = Trainer(deterministic=True, max_epochs=30, callbacks=[ErrorTracker()], logger=mlr_logger)
    mlr_trainer.fit(mlr_module, datamodule=mnist_dm)
    print(mlr_trainer.test(mlr_module, datamodule=mnist_dm))

if __name__ == '__main__':
    main()