import numpy as np
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                         summarize,
                         poly)
from sklearn.model_selection import train_test_split
from functools import partial
from sklearn.model_selection import \
    (cross_validate, KFold, ShuffleSplit)
from sklearn.base import clone
from ISLP.models import sklearn_sm

Auto = load_data('Auto')
cv_error = np.zeros(5)
Y = Auto['mpg']
H = np.array(Auto['horsepower'])
M = sklearn_sm(sm.OLS)
cv = KFold(n_splits=10, shuffle=True, random_state=0)
for i, d in enumerate(range(1,6)):
    X = np.power.outer(H, np.arange(d + 1))
    M_CV = cross_validate(M, X, Y, cv=cv)
    cv_error[i] = np.mean(M_CV['test_score'])

print(cv_error)

hp_model = sklearn_sm(sm.OLS, MS(['horsepower']))
validation = ShuffleSplit(n_splits=10, test_size=196, random_state=0)
results = cross_validate(hp_model, Auto.drop(['mpg'], axis=1), Auto['mpg'], cv=validation)
#print(results['test_score'])
print(results['test_score'].mean(), results['test_score'].std())