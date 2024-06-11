import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm

from statsmodels.stats.outliers_influence \
    import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm

from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)
import matplotlib.pyplot as plt

Boston = load_data("Boston")
X = MS([poly('lstat', degree=2), 'age']).fit_transform(Boston)
y = Boston['medv']
polyModel = sm.OLS(y, X)
polyResults = polyModel.fit()
print(summarize(polyResults))
print(polyResults.rsquared)

terms = Boston.columns.drop(['medv', 'age', 'indus'])
design = MS(terms)
X = design.fit_transform(Boston)

y = Boston['medv']
linearModel = sm.OLS(y, X)
linearResults = linearModel.fit()
print(anova_lm(linearResults, polyResults))
print(linearResults.rsquared)

ax = subplots(figsize=(8,8))[1]
ax.scatter(polyResults.fittedvalues, polyResults.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--')
plt.show()