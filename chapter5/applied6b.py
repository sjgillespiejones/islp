import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,summarize)
import matplotlib.pyplot as plt
import seaborn as sns

from ISLP import confusion_table
from ISLP.models import contrast
from sklearn.discriminant_analysis import \
    (LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from patsy.highlevel import dmatrices
import statsmodels.formula.api as smf

Default = load_data('Default')

Default['default_yes'] = (Default['default'] == 'Yes').astype('int')
x = Default[['income', 'balance']]
y = Default['default_yes']

np.random.seed(0)

def boot_fn(dataframe):
    model = smf.glm('default_yes ~ income + balance', data=dataframe, family=sm.families.Binomial()).fit()
    coef_income = model.params[1]
    coef_balance = model.params[2]
    return [coef_income, coef_balance]

print(boot_fn(Default))

def bootstrap(dataframe, sample_size=None):
    if(sample_size == None):
        sample_size = len(dataframe)

    bootSample_i = (np.random.rand(sample_size) * len(dataframe)).astype(int)
    bootSample_i = np.array(bootSample_i)
    bootSample_dataframe = dataframe.iloc[bootSample_i]
    return bootSample_dataframe

coefficients = []
n = 100
for i in range(0, n):
    coef_i = boot_fn(bootstrap(Default))
    coefficients.append(coef_i)

print(pd.DataFrame(coefficients).mean())