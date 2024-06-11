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

Auto = load_data('Auto')

y = Auto['mpg']
x = Auto['horsepower']
X = pd.DataFrame({ 'intercept': np.ones(Auto.shape[0]), 'horsepower' : x})
model = sm.OLS(y, X)
results = model.fit()

prediction = results.get_prediction([1, 98])
print(prediction.conf_int())

_, axes = subplots(1, 3)
ax = axes[0]
ax.scatter(x, y)
ax.set_xlabel('horsepower')
ax.set_ylabel('mpg')
ax.plot(x, results.fittedvalues, color='red')

ax2 = axes[1]
ax2.scatter(results.fittedvalues, results.resid, color='green')
ax2.axhline(0, c='k', ls='--')
print(results.summary())

infl = results.get_influence()
_, ax3 = subplots(figsize=(8,8))
ax3.set_xlabel('Index')
ax3.set_ylabel('Leverage')
ax3.scatter(np.arange(X.shape[0]), infl.hat_matrix_diag)
plt.show()
