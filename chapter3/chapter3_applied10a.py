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

Carseats = load_data('Carseats').dropna()

model = sm.formula.ols(formula='Sales ~ Price + US', data=Carseats)
results = model.fit()
print(results.summary())

print(results.conf_int())

_, ax = subplots(figsize=(8,8))
ax.scatter(results.fittedvalues, results.resid, color='green')
ax.axhline(0, c='k', ls='--')
plt.show()

_,ax2 = subplots(figsize=(8,8))
ax2.set_xlabel('Index')
ax2.set_ylabel('Leverage')
ax2.scatter(np.arange(Carseats.shape[0]), results.get_influence().hat_matrix_diag)
plt.show()

influence_results = results.get_influence().hat_diag_factor
keep_index = [i if x < 0.025 else -1 for (i, x) in enumerate(influence_results)]
print(keep_index)

ClippedData = Carseats[Carseats.index.isin(keep_index)]
updatedModel = sm.formula.ols(formula='Sales ~ Price + US', data=ClippedData)
updatedResults = updatedModel.fit()
print(updatedResults.summary())

print(updatedResults.conf_int())

_,ax3 = subplots(figsize=(8,8))
ax3.set_xlabel('Index')
ax3.set_ylabel('Leverage')
ax3.scatter(np.arange(ClippedData.shape[0]), updatedResults.get_influence().hat_matrix_diag)
plt.show()