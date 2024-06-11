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

def abline(ax, intercept, slope, *args, **kwargs):
    xlim = ax.get_xlim()
    ylim = [slope * xlim[0] + intercept, slope * xlim[1] + intercept]
    ax.plot(xlim, ylim, *args, **kwargs)

Boston = load_data("Boston")
design = MS(['lstat'])
X = design.fit_transform(Boston)

y = Boston['medv']
model = sm.OLS(y, X)
results = model.fit()

new_df = pd.DataFrame({'lstat': [5,10,15]})
newX = design.transform(new_df)

new_predictions = results.get_prediction(newX)

ax = Boston.plot.scatter('lstat', 'medv')
abline(
    ax,
    results.params[0],
    results.params[1],
    'r--',
    linewidth=3
)
plt.show()