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

rng = np.random.default_rng(10)
x1 = rng.uniform(0, 1, size=100)
x2 = 0.5 * x1 + rng.normal(size=100) / 10
y = 2 + 2 * x1 + 0.3 * x2 + rng.normal(size=100)

x1 = np.concatenate([x1, [0.1]])
x2 = np.concatenate([x2, [0.8]])
y = np.concatenate([y, [6]])

_, ax = subplots(figsize=(8,8))
ax.scatter(x2, y)
plt.show()

print(np.corrcoef(x1, x2))

dataframe = pd.DataFrame(list(zip(x1, x2, y)), columns=['x1', 'x2', 'y'])
model = sm.formula.ols(formula='y ~ x1 + x2', data=dataframe)
results = model.fit()
print(results.summary())

_, ax = subplots(figsize=(8,8))
ax.scatter(results.fittedvalues, results.resid, color='green')
ax.axhline(0, c='k', ls='--')
plt.show()

# _,ax3 = subplots(figsize=(8,8))
# ax3.set_xlabel('Index')
# ax3.set_ylabel('Leverage')
# ax3.scatter(np.arange(y.shape[0]), results.get_influence().hat_matrix_diag)
# plt.show()