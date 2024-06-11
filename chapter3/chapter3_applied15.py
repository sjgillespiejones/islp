import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm

from statsmodels.stats.outliers_influence \
    import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm

from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)
import matplotlib.pyplot as plt

Boston = load_data('Boston').dropna()

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(Boston.head())

partialModel = sm.formula.ols(formula='crim ~ medv', data=Boston)
partialResults = partialModel.fit()
print(partialResults.summary())

fullModel = sm.formula.ols(formula='crim ~ zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + lstat + medv', data=Boston)
fullResults = fullModel.fit()



univariateCoefficients = [-0.0739, 0.5098, -1.8928, 31.2485, -2.6841, 0.1078, -1.5509, 0.6179, 0.0297, 1.1520, 0.5488, -0.3632]
multivariateCoefficients = [0.0457, -0.0584, -0.8254, -9.9576, 0.6289, -0.0008, -1.0122, 0.6125, -0.0038, -0.3041, 0.1388, -0.2201]

# _, ax = subplots(figsize=(8,8))
# ax.scatter(univariateCoefficients, multivariateCoefficients)
# plt.show()

# faster way
print("{:>9} {:>22} {:>24}".format("predictor", "coef","pvalue"))
coefs = {}

predictors = [c for c in list(Boston) if c not in ["crim"]]
for predictor in predictors:
    model = 'crim ~ ' + predictor
    res = sm.formula.ols(formula = model, data=Boston).fit()
    # http://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.RegressionResults.html
    print("{:>9} {:>22} {:>24}".format(predictor, res.params[predictor],res.pvalues[predictor]))
    coefs[predictor] = [res.params[predictor]]

plt.figure(figsize=(20, 20))

for i, predictor in enumerate(predictors):
    model = 'crim ~ ' + predictor + " + np.power(" + predictor + ", 2) + np.power(" + predictor + ", 3)"
    res = sm.formula.ols(formula = model, data=Boston).fit()
    plt.subplot(5, 3, i + 1)
    plt.xlabel(predictor)
    plt.ylabel("CRIM")
    plt.scatter(Boston[predictor], Boston['crim'])
    x = np.linspace(min(Boston[predictor]), max(Boston[predictor]), 100)
    y = res.params[0] + x * res.params[1] + res.params[2] * (x ** 2) + res.params[3] * (x ** 3)
    plt.plot(x, y, color='red')

plt.show()