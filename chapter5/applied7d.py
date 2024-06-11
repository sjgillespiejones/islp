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

Weekly = load_data('Weekly')
Weekly['Direction_up'] = (Weekly['Direction'] == 'Up').astype(int)

X = Weekly[['Lag1', 'Lag2']]
y = Weekly['Direction_up']

# logit = LogisticRegression().fit(X, y)
# probs = logit.predict(X)
#
# LOOCV_logit = LogisticRegression().fit(X.iloc[1:], y.iloc[1:])
#
# print(LOOCV_logit.predict([X.iloc[0]]))
# print(y[0])

n = len(X)
errors = np.zeros(n)

for i in range(n):
    subset = ~X.index.isin([i])
    model = LogisticRegression().fit(X[subset], y[subset])
    hasError = model.predict([X.iloc[i]]) != y[i]
    if(hasError):
        errors[i] = 1

print(np.mean(errors))