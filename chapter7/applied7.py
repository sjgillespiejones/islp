import numpy as np, pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (summarize,
                         poly,
                         ModelSpec as MS)
from statsmodels.stats.anova import anova_lm

from pygam import (s as s_gam,
                   l as l_gam,
                   f as f_gam,
                   LinearGAM, LogisticGAM)
from ISLP.transforms import (BSpline, NaturalSpline)
from ISLP.models import bs, ns
from ISLP.pygam import (approx_lam, degrees_of_freedom,
                        plot as plot_gam, anova as anova_gam)
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

Wage = load_data('Wage')
#print(Wage['maritl'].cat.categories)
categories = ['maritl', 'race', 'region', 'jobclass', 'health', 'health_ins']
for col in categories:
    Wage[col] = Wage[col].astype('category')