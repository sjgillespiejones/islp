import numpy as np
import pandas as pd
import sklearn.linear_model
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
from ISLP.models import ModelSpec as MS
from functools import partial
import statsmodels.api as sm
from statsmodels.api import OLS
from ISLP.models import \
(Stepwise , sklearn_selected , sklearn_selection_path)
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, LassoCV
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(1)

x = rng.normal(size=100)
epsilon = rng.normal(size=100)
betaZero = 5
betaOne = 2
betaTwo = 3.5
betaThree = -2.22

y = betaZero + (betaOne * x) + (betaTwo * (x ** 2)) + (betaThree * (x ** 3)) + epsilon
dataframe = pd.DataFrame({'y': y, 'x': x, 'x_squared': x ** 2, 'x_cubed': x ** 3, 'x4': x**4, 'x5': x**5, 'x6': x**6, 'x7': x**7, 'x8': x**8, 'x9': x**9, 'x10': x**10})
X = dataframe[['x', 'x_squared', 'x_cubed', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]
Y = dataframe[['y']]
print(dataframe)

forward_selector = SequentialFeatureSelector(LinearRegression(), k_features=5, forward=True, floating=False, scoring='neg_mean_squared_error', cv=5)
forward_selector.fit(X, Y)
print(forward_selector.k_feature_names_)
print(forward_selector.k_score_)

forwardSelectionModel = sm.formula.ols(formula='y ~ x + x_squared + x_cubed + x4 + x10', data=dataframe)
forwardSelectionResults = forwardSelectionModel.fit()
print(forwardSelectionResults.summary())

backward_selector = SequentialFeatureSelector(LinearRegression(), k_features=5, forward=False,  floating=False, scoring='neg_mean_squared_error', cv=5)
backward_selector.fit(X, Y)
print(backward_selector.k_feature_names_)
print(backward_selector.k_score_)

backwardsSelectionModel = sm.formula.ols(formula='y ~ x + x_squared + x_cubed + x6 + x8', data=dataframe)
backwardsSelectionResults = backwardsSelectionModel.fit()
print(backwardsSelectionResults.summary())

lasso_cv = LassoCV(alphas=[0.00001,0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6,7,8,9, 10, 15, 20, 50, 100]).fit(X, Y)
print(lasso_cv.score(X, Y))
print(lasso_cv.coef_)

print(list(zip(lasso_cv.coef_, X)))

fig, ax = subplots(figsize=(12,12))
ax.plot(
    lasso_cv.alphas_,
    lasso_cv.mse_path_.mean(axis=-1),
    "k",
    label="Average across the folds",
    linewidth=2
)
ax.axvline(lasso_cv.alpha_, linestyle="--", color="g", label="alpha: CV estimate")
ax.set_xscale('log')
ax.legend()
ax.set_xlabel('alphas')
ax.set_ylabel('Mean square error')
ax.set_title('Mean square error on each fold')
ax.axis('tight')
plt.show()

