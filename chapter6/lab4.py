import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
from statsmodels.api import OLS
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from functools import partial

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from ISLP.models import \
(Stepwise , sklearn_selected , sklearn_selection_path)
from l0bnb import fit_path
import matplotlib.pyplot as plt

Hitters = load_data('Hitters')
Hitters = Hitters.dropna()
Y = np.array(Hitters['Salary'])
design = MS(Hitters.columns.drop('Salary')).fit(Hitters)

D = design.fit_transform(Hitters)
D = D.drop('intercept', axis=1)
X = np.asarray(D)

path = fit_path(X, Y, max_nonzeros=X.shape[1])
print(path[3])