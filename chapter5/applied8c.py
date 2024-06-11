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

rng = np.random.default_rng(2)
x = rng.normal(size=100)
x_squared = -2 * x ** 2
epsilon = rng.normal(size=100)
y = x + x_squared + epsilon

dataframe = pd.DataFrame({ 'y': y, 'x': x, 'x_squared': x_squared, 'epsilon': epsilon})

# print(dataframe.head())
# result = sm.formula.ols(formula='y ~ x', data=dataframe).fit()
#
# print(result.predict(dataframe.iloc[0]))

index_to_remove = [0]



def leave_one_out_cross_validate(frame, formula):
    n = len(frame)
    squared_errors = np.zeros(n)
    for i in range(n):
        subset = frame[~frame.index.isin(index_to_remove)]
        result = sm.formula.ols(formula=formula, data=subset).fit()
        prediction = result.predict(frame.iloc[i])
        actual = frame.iloc[i]['y']
        squared_errors[i] = np.square(prediction - actual)
    return squared_errors

print(np.mean(leave_one_out_cross_validate(dataframe, 'y ~ x + epsilon')))
print(np.mean(leave_one_out_cross_validate(dataframe, 'y ~ x + np.square(x) + epsilon')))
print(np.mean(leave_one_out_cross_validate(dataframe, 'y ~ x + np.square(x) + np.power(x, 3) + epsilon')))
print(np.mean(leave_one_out_cross_validate(dataframe, 'y ~ x + np.square(x) + np.power(x, 3) + np.power(x, 4) + epsilon')))