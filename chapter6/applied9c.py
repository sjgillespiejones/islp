import numpy as np
import pandas as pd
from ISLP import load_data
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

College = load_data('College')

Y = College['Apps']
X = College.drop(['Apps', 'Private'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

lambdas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10_000, 100_000]
K = 5
kfold = skm.KFold(K, random_state=0,shuffle=True)
ridgeCV = skl.ElasticNetCV(alphas=lambdas, l1_ratio=0, cv=kfold)
scaler = StandardScaler(with_mean=True, with_std=True).fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridgeCV.fit(X_train_scaled, y_train)
Rsquared = ridgeCV.score(X_test_scaled, y_test)
mse = np.mean((ridgeCV.predict(X_test_scaled) - y_test) ** 2)
print('Ridge R^2: ', Rsquared)
print('Ridge MSE: ', mse)

lassoCV = skl.LassoCV(alphas=lambdas, cv=kfold)
lassoCV.fit(X_train_scaled, y_train)
Rsquared = lassoCV.score(X_test_scaled, y_test)
mse = np.mean((lassoCV.predict(X_test_scaled) - y_test) ** 2)
print('Lasso R^2: ', Rsquared)
print('Lasso MSE: ', mse)
print('Lasso coefficients: ', list(zip(lassoCV.coef_, X)))
print('Lasso alpha val: ', lassoCV.alpha_)

pca = PCA(n_components=2)
linreg = skl.LinearRegression()
pipe = Pipeline([('scaler', scaler), ('pca', pca), ('linreg', linreg)])
pipe.fit(X_train, y_train)

param_grid = {'pca__n_components': range(1, 20)}
grid = skm.GridSearchCV(pipe, param_grid, cv=kfold, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

pcr_mse = np.mean((grid.predict(X_test) - y_test) ** 2)
print('PCR MSE: ', pcr_mse)
print('Best M: ', grid.best_index_)

pls = PLSRegression(n_components=2, scale=True)
pls.fit(X_train, y_train)
param_grid = {'n_components': range(1,20)}
plsGrid = skm.GridSearchCV(pls, param_grid, cv=kfold, scoring='neg_mean_squared_error')
plsGrid.fit(X_train, y_train)

pls_mse = np.mean((plsGrid.predict(X_test) - y_test) ** 2)
print('PLS MSE: ', pls_mse)
print('Best M: ', plsGrid.best_index_)