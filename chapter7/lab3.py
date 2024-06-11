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

from chapter7.lab1 import plot_wage_fit

Wage = load_data('Wage')
y = Wage['wage']
age = Wage['age']

age_grid = np.linspace(age.min(), age.max(), 100)
age_df = pd.DataFrame({'age': age_grid})

# bs_ = BSpline(internal_knots=[25, 40, 60], intercept=True).fit(age)
# bs_age = bs_.transform(age)
# print(bs_age.shape)

# bs_age = MS([bs('age', internal_knots=[25, 40, 60], name='bs(age, knots)')])
# Xbs = bs_age.fit_transform(Wage)
# M = sm.OLS(y, Xbs).fit()
# print(summarize(M))

#print(BSpline(df=6).fit(age).internal_knots_)
# bs_age0 = MS([bs('age', df=3, degree=0)]).fit(Wage)
# Xbs0 = bs_age0.transform(Wage)
# print(summarize(sm.OLS(y, Xbs0).fit()))

ns_age = MS([ns('age', df=5)]).fit(Wage)
M_ns = sm.OLS(y, ns_age.transform(Wage)).fit()
print(summarize(M_ns))

plot_wage_fit(age_df, ns_age, 'Natural spline, df=5')
plt.show()