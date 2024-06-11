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

Hitters = load_data('Hitters')
#print(np.isnan(Hitters['Salary']).sum())
Hitters = Hitters.dropna()
print(Hitters.shape)

def nCp(sigma2, estimator, X, Y):
    "Negative Cp statistic"
    n, p = X.shape
    Yhat = estimator.predict(X)
    RSS = np.sum((Y - Yhat) ** 2)
    return -1 * (RSS + 2 * p * sigma2) / 2

design = MS(Hitters.columns.drop('Salary')).fit(Hitters)
Y = np.array(Hitters['Salary'])
X = design.transform(Hitters)
sigma2 = OLS(Y, X).fit().scale

neg_Cp = partial(nCp, sigma2)

strategy = Stepwise.first_peak(design, direction='forward', max_terms=len(design.terms))

hitters_MSE = sklearn_selected(OLS, strategy)
hitters_MSE.fit(Hitters, Y)
#print(hitters_MSE.selected_state_)

hitters_Cp = sklearn_selected(OLS, strategy, scoring=neg_Cp)
hitters_Cp.fit(Hitters, Y)
print(hitters_Cp.selected_state_)