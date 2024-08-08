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

class IMDBModel(nn.Module):

    def __init__(self, input_size):
        super(IMDBModel, self).__init__()
        self.dense1 = nn.Linear(input_size, 16)
        self.activation = nn.ReLU()
        self.dense2 = nn.Linear(16, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        val = x
        for _map in [self.dense1,
                     self.activation,
                     self.dense2,
                     self.activation,
                     self.output]:
            val = _map(val)
        return torch.flatten(val)

def main():
    seed_everything(0, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # (imdb_seq_train, imdb_seq_test) = load_sequential(root='data/IMDB')
    # padded_sample = np.asarray(imdb_seq_train.tensors[0][0])
    # sample_review = padded_sample[padded_sample > 0][:12]
    # print(sample_review[:12])

    # lookup = load_lookup(root='data/IMDB')
    # print(' '.join(lookup[i] for i in sample_review))

    max_num_workers = 10
    (imdb_train, imdb_test) = load_tensor(root='data/IMDB')
    imdb_dm = SimpleDataModule(imdb_train, imdb_test, validation=2000, num_workers=min(6, max_num_workers), batch_size=512)

    imdb_model = IMDBModel(imdb_test.tensors[0].size()[1])
    summary(imdb_model, input_size=imdb_test.tensors[0].size(), col_names=['input_size', 'output_size', 'num_params'])

    imdb_optimizer = RMSprop(imdb_model.parameters(), lr=0.001)
    imdb_module = SimpleModule.binary_classification(imdb_model, optimizer=imdb_optimizer)
    imdb_logger = CSVLogger('logs', name='IMDB')
    imdb_trainer = Trainer(deterministic=True,
                           max_epochs=30,
                           logger=imdb_logger,
                           callbacks=[ErrorTracker()])
    imdb_trainer.fit(imdb_module,
                     datamodule=imdb_dm)

    test_results = imdb_trainer.test(imdb_module, datamodule=imdb_dm)
    print(test_results)




if __name__ == '__main__':
    main()