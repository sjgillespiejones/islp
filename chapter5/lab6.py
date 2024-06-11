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

from chapter5.lab5 import boot_SE

Auto = load_data('Auto')

def boot_OLS(model_matrix, response, D, idx):
    D_ = D.loc[idx]
    Y_ = D_[response]
    X_ = clone(model_matrix).fit_transform(D_)
    return sm.OLS(Y_, X_).fit().params

hp_func = partial(boot_OLS, MS(['horsepower']), 'mpg')

#rng = np.random.default_rng(0)
#array = np.array([hp_func(Auto, rng.choice(392, 392, replace=True)) for _ in range(10)])
#print(array)

hp_se = boot_SE(hp_func, Auto, B=1000, seed=10)
#print(hp_se)

hp_model = sklearn_sm(sm.OLS, MS(['horsepower']))
hp_model.fit(Auto, Auto['mpg'])
model_se = summarize(hp_model.results_)['std err']
#print(model_se)