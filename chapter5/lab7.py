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
from chapter5.lab6 import boot_OLS

Auto = load_data('Auto')
quad_model = MS([poly('horsepower', 2, raw=True)])
quad_func = partial(boot_OLS, quad_model, 'mpg')
print(boot_SE(quad_func, Auto, B=1000))

M = sm.OLS(Auto['mpg'], quad_model.fit_transform(Auto))
print(summarize(M.fit())['std err'])