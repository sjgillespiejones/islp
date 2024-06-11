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

rng = np.random.default_rng(1)
x = rng.normal(size=100)
y = 2 * x + rng.normal(size=100)
dataframe = pd.DataFrame(list(zip(x, y)), columns=['x', 'y'])
model = sm.formula.ols(formula='x ~ y', data=dataframe)
results = model.fit()
print(results.summary())

_, ax = subplots(figsize=(8,8))
ax.scatter(y, x, color='green')
ax.plot(y, results.fittedvalues, color='red')
plt.show()