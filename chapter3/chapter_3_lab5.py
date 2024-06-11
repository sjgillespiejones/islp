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
X = MS(['lstat', 'age', ('lstat', 'age')]).fit_transform(Boston)
y = Boston['medv']
model = sm.OLS(y, X)
results = model.fit()
print(summarize(results))
print(results.rsquared)