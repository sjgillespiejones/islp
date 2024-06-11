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

Portfolio = load_data('Portfolio')


def alpha_func(D, idx):
    cov_ = np.cov(D[['X', 'Y']].loc[idx], rowvar=False)
    return ((cov_[1, 1] - cov_[0, 1]) / (cov_[0, 0] + cov_[1, 1] - 2 * cov_[0, 1]))


# print(alpha_func(Portfolio, range(100)))
#rng = np.random.default_rng(0)
#print(alpha_func(Portfolio, rng.choice(100, 100, replace=True)))


def boot_SE(func, D, n=None, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    first_, second_ = 0, 0
    n = n or D.shape[0]
    for _ in range(B):
        idx = rng.choice(D.index, n, replace=True)
        value = func(D, idx)
    first_ += value
    second_ += value ** 2
    return np.sqrt(second_ / B - (first_ / B) ** 2)


alpha_SE = boot_SE(alpha_func, Portfolio, B=1000, seed=0)
#print(alpha_SE)
