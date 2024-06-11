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

Xs = X - X.mean(0)[None, :]
X_scale = X.std(0)
Xs = Xs / X_scale[None, :]
lambdas = 10 ** np.linspace(8, -2, 100) / Y.std()
soln_array = skl.ElasticNet.path(Xs, Y, l1_ratio = 0., alphas=lambdas)[1]
#print(soln_array.shape)
soln_path = pd.DataFrame(soln_array.T, columns=D.columns, index=-np.log(lambdas))
soln_path.index.name = 'negative log(lambda)'
#print(soln_path)

path_fig, ax = subplots(figsize=(8,8))
soln_path.plot(ax=ax, legend=False)
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)
ax.legend(loc='upper left')
plt.show()

#beta_hat = soln_path.loc[soln_path.index[59]]
#print(lambdas[59], beta_hat)
#print(np.linalg.norm(beta_hat))
ridge = skl.ElasticNet(alpha=lambdas[59], l1_ratio=0)
scaler = StandardScaler(with_mean=True, with_std=True)
pipe = Pipeline(steps=[('scaler', scaler), ('ridge', ridge)])
pipe.fit(X, Y)

#print(np.linalg.norm(ridge.coef_))

validation = skm.ShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
ridge.alpha = 0.01
results = skm.cross_validate(ridge, X, Y, scoring='neg_mean_squared_error', cv=validation)
#print(-results['test_score'])

ridge.alpha = 1e10
results = skm.cross_validate(ridge, X, Y, scoring='neg_mean_squared_error', cv=validation)
#print(-results['test_score'])

param_grid = {'ridge__alpha': lambdas}
grid = skm.GridSearchCV(pipe, param_grid, cv=validation, scoring='neg_mean_squared_error')
grid.fit(X, Y)
#print(grid.best_params_['ridge__alpha'])
#print(grid.best_estimator_)

kfold = skm.KFold(5, random_state=0,shuffle=True)

grid = skm.GridSearchCV(pipe, param_grid, cv=kfold, scoring='neg_mean_squared_error')
grid.fit(X, Y)
print(grid.best_params_['ridge__alpha'])
print(grid.best_estimator_)

grid_r2 = skm.GridSearchCV(pipe, param_grid, cv=kfold)
grid_r2.fit(X, Y)
r2_fig, ax = subplots(figsize=(8,8))
ax.errorbar(
    -np.log(lambdas),
    grid_r2.cv_results_['mean_test_score'],
    yerr=grid_r2.cv_results_['std_test_score'] / np.sqrt(5)
)
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Cross-validated $R^2$', fontsize=20)
plt.show()