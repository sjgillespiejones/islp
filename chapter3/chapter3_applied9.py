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

Auto = load_data("Auto")
pd.plotting.scatter_matrix(Auto)
plt.show()

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(Auto.corr())

