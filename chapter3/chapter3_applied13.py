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
x = rng.normal(scale=1, size=100)
eps = np.random.normal(scale=0.375, size=100)
Y = -1 + (x / 2) + eps

_, ax = subplots(figsize=(8,8))
ax.scatter(Y, x, color='green')


dataframe = pd.DataFrame(list(zip(x, Y)), columns=['x', 'y'])
model = sm.formula.ols(formula='Y ~ x', data=dataframe)
results = model.fit()
print(results.summary())

fit, = ax.plot(results.fittedvalues, x, color='red')
xModel = np.linspace(dataframe.x.min(), dataframe.x.max(), 100)
ymodel = -1 + 0.5 * xModel
modelLine, = plt.plot(ymodel, xModel, color='blue')
plt.legend([fit, modelLine], ['Fit line', 'Model line'])
plt.show()