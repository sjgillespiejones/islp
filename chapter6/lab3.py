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
strategy = Stepwise.fixed_steps(design, len(design.terms), direction='forward')
full_path = sklearn_selection_path(OLS, strategy)
full_path.fit(Hitters, Y)

Yhat_in = full_path.predict(Hitters)

K = 5
kfold = skm.KFold(K, random_state=0,shuffle=True)
Yhat_cv = skm.cross_val_predict(full_path, Hitters, Y, cv=kfold)
#print(Yhat_cv.shape)

cv_mse = []
for train_idx, test_idx in kfold.split(Y):
    errors = (Yhat_cv[test_idx] - Y[test_idx, None]) ** 2
    cv_mse.append(errors.mean(0))

cv_mse = np.array(cv_mse).T
print(cv_mse)

validation = skm.ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

for train_idx, test_idx in validation.split(Y):
    full_path.fit(Hitters.iloc[train_idx], Y[train_idx])
    Yhat_val = full_path.predict(Hitters.iloc[test_idx])
    errors = (Yhat_val - Y[test_idx, None]) ** 2
    validation_mse = errors.mean(0)

mse_fig, ax = subplots(figsize=(8,8))
insample_mse = ((Yhat_in - Y[:, None]) ** 2).mean(0)
n_steps = insample_mse.shape[0]
ax.plot(np.arange(n_steps),
        insample_mse,
        'k',
        label='In-sample')
ax.set_ylabel('MSE', fontsize=20)
ax.set_xlabel('# steps of forward stepwise', fontsize=20)

ax.errorbar(np.arange(n_steps), cv_mse.mean(1), cv_mse.std(1) / np.sqrt(K), label='Cross-validated', c='r')
ax.plot(np.arange(n_steps), validation_mse, 'b--', label='Validation')
ax.legend()
ax.set_xticks(np.arange(n_steps)[::2])
ax.set_ylim([50_000, 250_000])
plt.show()