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

Auto = load_data('Auto').dropna()
model = sm.formula.ols(formula='mpg ~ cylinders + displacement + horsepower + weight + acceleration + year + origin', data=Auto)
results = model.fit()
print(results.summary())
print(anova_lm(results, typ=1))

_, ax = subplots(figsize=(8,8))
ax.scatter(results.fittedvalues, results.resid, color='green')
ax.axhline(0, c='k', ls='--')
plt.show()

_,ax2 = subplots(figsize=(8,8))
ax2.set_xlabel('Index')
ax2.set_ylabel('Leverage')
ax2.scatter(np.arange(Auto.shape[0]), results.get_influence().hat_matrix_diag)
plt.show()

