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
model = sm.formula.ols(formula='mpg ~ np.sqrt(horsepower) + np.square(weight) + weight + horsepower + year', data=Auto)
results = model.fit()
print(results.summary())
print(anova_lm(results, typ=1))

_, ax = subplots(figsize=(8,8))
ax.scatter(results.fittedvalues, results.resid, color='green')
ax.axhline(0, c='k', ls='--')
plt.show()