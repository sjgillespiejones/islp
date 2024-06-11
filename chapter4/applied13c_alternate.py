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

Weekly = load_data('Weekly')

train_cols = ['Lag1','Lag2','Lag3','Lag4','Lag5','Volume']
lr = LogisticRegression()
mod = lr.fit(Weekly[train_cols], Weekly['Direction'])
Weekly['Direction'] = Weekly['Direction'].map({'Down': 0, 'Up': 1})
y, X = dmatrices('Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume', data=Weekly, return_type='dataframe')
logit = sm.Logit(y, X)
result = logit.fit()
print(result.summary())
