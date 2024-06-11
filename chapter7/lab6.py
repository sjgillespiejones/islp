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

Wage = load_data('Wage')
y = Wage['wage']
age = Wage['age']
year = Wage['year']
education = Wage['education']
high_earn = Wage['high_earn'] = y > 250

gam_full = LinearGAM(s_gam(0) + s_gam(1, n_splines=7) + f_gam(2, lam=0))
Xgam = np.column_stack([age, year, education.cat.codes])
gam_full = gam_full.fit(Xgam, y)

age_term = gam_full.terms[0]
age_term.lam = approx_lam(Xgam, age_term, df=4+1)
year_term = gam_full.terms[1]
year_term.lam = approx_lam(Xgam, year_term, df=4+1)
gam_full = gam_full.fit(Xgam, y)

gam_0 = LinearGAM(age_term + f_gam(2, lam=0))
gam_0.fit(Xgam, y)
gam_linear = LinearGAM(age_term + l_gam(1, lam=0) + f_gam(2, lam=0))
gam_linear.fit(Xgam, y)

#print(anova_gam(gam_0, gam_linear, gam_full))

gam_0 = LinearGAM(year_term + f_gam(2, lam=0))
gam_linear = LinearGAM(l_gam(0, lam=0) + year_term + f_gam(2, lam=0))
gam_0.fit(Xgam, y)
gam_linear.fit(Xgam, y)
#print(anova_gam(gam_0, gam_linear, gam_full))
#print(gam_full.summary())

Yhat = gam_full.predict(Xgam)

gam_logit = LogisticGAM(age_term + l_gam(1, lam=0) + f_gam(2, lam=0))
gam_logit.fit(Xgam, high_earn)

# fig, ax = subplots(figsize=(8,8))
# ax = plot_gam(gam_logit, 2)
# ax.set_xlabel('Education')
# ax.set_ylabel('Effect on wage')
# ax.set_title('Partial dependence of wage on education',fontsize=20)
# ax.set_xticklabels(Wage['education'].cat.categories, fontsize=8)
# plt.show()

#print(pd.crosstab(high_earn, education))

only_hs = education == '1. < HS Grad'
Wage_ = Wage.loc[~only_hs]
Xgam_ = np.column_stack([Wage_['age'], Wage_['year'], Wage_['education'].cat.codes-1])
high_earn_ = Wage_['high_earn']

gam_logit_ = LogisticGAM(age_term + year_term + f_gam(2, lam=0))
gam_logit_.fit(Xgam_, high_earn_)
fig, ax = subplots(figsize=(8,8))
ax = plot_gam(gam_logit_, 0)
ax.set_xlabel('Age')
ax.set_ylabel('Effect on wage')
ax.set_title('Partial dependence of high earner status on age',fontsize=20)
#ax.set_xticklabels(Wage['education'].cat.categories[1:], fontsize=8)
plt.show()

age_grid = np.linspace(age.min(), age.max(), 100)

lowess = sm.nonparametric.lowess
fig, ax = subplots(figsize=(8,8))
ax.scatter(age, y, facecolor='gray', alpha=0.05)
for span in [0.2, 0.5]:
    fitted = lowess(y, age, frac=span, xvals=age_grid)
    ax.plot(age_grid, fitted, label='{:.1f}'.format(span), linewidth=4)
ax.set_xlabel('Age', fontsize=20)
ax.set_ylabel('Wage', fontsize=20); ax.legend(title='span', fontsize=15)
ax.legend(title='span', fontsize=15)
plt.show()