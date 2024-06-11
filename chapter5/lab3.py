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
H = np.array(Auto['horsepower'])
Y = Auto['mpg']
M = sklearn_sm(sm.OLS)
for i, d in enumerate(range(1, 6)):
    X = np.power.outer(H, np.arange(d + 1))
    M_CV = cross_validate(M, X, Y, cv=Auto.shape[0])
    cv_error[i] = np.mean(M_CV['test_score'])

print(cv_error)