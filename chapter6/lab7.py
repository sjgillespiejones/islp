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
lambdas = 10 ** np.linspace(8, -2, 100) / Y.std()
K = 5
kfold = skm.KFold(K, random_state=0,shuffle=True)
scaler = StandardScaler(with_mean=True, with_std=True)
Xs = X - X.mean(0)[None, :]
X_scale = X.std(0)
Xs = Xs / X_scale[None, :]

lassoCV = skl.ElasticNetCV(n_alphas=100, l1_ratio=1, cv=kfold)
pipeCV = Pipeline(steps=[('scaler', scaler), ('lasso', lassoCV)])
pipeCV.fit(X, Y)
tuned_lasso = pipeCV.named_steps['lasso']
#print(tuned_lasso.alpha_)

lambdas, soln_array = skl.Lasso.path(Xs, Y, l1_ratio=1, n_alphas=100)[:2]
soln_path = pd.DataFrame(soln_array.T, columns=D.columns, index=-np.log(lambdas))
#print(soln_path)

path_fig, ax = subplots(figsize=(12,12))
soln_path.plot(ax=ax, legend=False)
ax.legend(loc='upper left')
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)
plt.show()

print(np.min(tuned_lasso.mse_path_.mean(1)))


lassoCV_fig, ax = subplots(figsize=(12,12))
ax.errorbar(
    -np.log(tuned_lasso.alphas_),
    tuned_lasso.mse_path_.mean(1),
    yerr=tuned_lasso.mse_path_.std(1) / np.sqrt(K)
)
ax.axvline(-np.log(tuned_lasso.alpha_), c='k', ls='--')
ax.set_ylim([50_000, 250_000])
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Cross-validated MSE', fontsize=20)
plt.show()
print(tuned_lasso.coef_)

inner_cv = skm.KFold(n_splits=5, shuffle=True, random_state=2)
lassoInnerCV = skl.LassoCV(alphas=lambdas, cv=inner_cv)
pipeCV = Pipeline(steps=[('scaler', scaler), ('lasso', lassoInnerCV)])
results = skm.cross_validate(
    pipeCV,
    X,
    Y,
    cv=inner_cv,
    scoring='neg_mean_squared_error'
)
print(-results['test_score'])