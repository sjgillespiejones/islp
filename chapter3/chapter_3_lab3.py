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

Boston = load_data("Boston")
design = MS(['lstat'])
X = design.fit_transform(Boston)

y = Boston['medv']
model = sm.OLS(y, X)
results = model.fit()

_, ax = subplots(figsize=(8,8))
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--')
plt.show()

infl = results.get_influence()
_, ax2 = subplots(figsize=(8,8))
ax2.scatter(np.arange(X.shape[0]), infl.hat_matrix_diag)
ax.set_xlabel('Index')
ax.set_ylabel('Leverage')
print(np.argmax(infl.hat_matrix_diag))
plt.show()