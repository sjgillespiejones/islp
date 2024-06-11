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

age_grid = np.linspace(age.min(), age.max(), 100)

X_age = np.asarray(age).reshape((-1, 1))
gam = LinearGAM(s_gam(0, lam=0.6))
gam.fit(X_age, y)

fig, ax = subplots(figsize=(8,8))
ax.scatter(age, y, facecolor='gray', alpha=0.5)
for lam in np.logspace(-2, 6, 5):
    gam = LinearGAM(s_gam(0, lam=lam)).fit(X_age, y)
    ax.plot(age_grid, gam.predict(age_grid), label='{:.1e}'.format(lam),linewidth=3)

ax.set_xlabel('Age', fontsize=20)
ax.set_ylabel('Wage', fontsize=20)
ax.legend(title='$\lambda$')

gam_opt = gam.gridsearch(X_age, y)
ax.plot(age_grid, gam_opt.predict(age_grid), label='Grid search', linewidth=4)
ax.legend()
plt.show()

age_term = gam.terms[0]
lam_4 = approx_lam(X_age, age_term, 4)
age_term.lam = lam_4
print(degrees_of_freedom(X_age, age_term))

fig, ax = subplots(figsize=(8,8))
ax.scatter(X_age, y, facecolor='gray', alpha=0.3)
for df in [1,3,4,8,15]:
    lam = approx_lam(X_age, age_term, df+1)
    age_term.lam = lam
    gam.fit(X_age, y)
    ax.plot(age_grid, gam.predict(age_grid), label='{:d}'.format(df), linewidth=4)
ax.set_xlabel('Age', fontsize=20)
ax.set_ylabel('Wage', fontsize=20)
ax.legend(title='Degrees of freedom')
plt.show()