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
y = Wage['wage']
age = Wage['age']

scores = []
cuts = range(1, 20)
for cut in cuts:
    cut_age = pd.qcut(age, cut)
    X = pd.get_dummies(cut_age)
    model = LinearRegression().fit(X, y)
    score = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
    scores.append(np.mean(score))

scores = np.abs(scores)

plt.plot(cuts, scores)
plt.ylabel('Mean squared error (CV)')
plt.xlabel('Cuts')
plt.xlim(0, 20)
plt.show()


# 13 cuts is best
best_cut = np.argmin(scores)
X = pd.get_dummies(pd.qcut(age, best_cut))
model = LinearRegression().fit(X, y)

X_lin = np.linspace(18, 80)
groups_aux = pd.qcut(X_lin, best_cut)
aux_dummies = pd.get_dummies(groups_aux)
y_lin = model.predict(aux_dummies)
plt.scatter(age, y)
plt.plot(X_lin, y_lin, '-r')
plt.xlabel('Age')
plt.ylabel('Wage')
plt.show()