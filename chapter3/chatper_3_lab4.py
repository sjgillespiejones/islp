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
terms = Boston.columns.drop(['medv', 'age', 'indus'])
design = MS(terms)
X = design.fit_transform(Boston)

y = Boston['medv']
model = sm.OLS(y, X)
results = model.fit()
vals = [VIF(X, i)
        for i in range(1, X.shape[1])]
vif = pd.DataFrame({'vif': vals}, index=X.columns[1:])
print(vif)
print(results.rsquared)