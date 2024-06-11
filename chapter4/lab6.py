import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,summarize)
import matplotlib.pyplot as plt

from ISLP import confusion_table
from ISLP.models import contrast
from sklearn.discriminant_analysis import \
    (LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

Smarket = load_data('Smarket')

model = MS(['Lag1', 'Lag2'], intercept=0).fit(Smarket)
X = model.transform(Smarket)
y = Smarket.Direction == 'Up'

train = (Smarket.Year < 2005)
X_train, X_test = X.loc[train], X.loc[~train]
y_train, y_test = y.loc[train], y.loc[~train]
D = Smarket.Direction
L_train, L_test = D.loc[train], D.loc[~train]

NB = GaussianNB()
NB.fit(X_train, L_train)
# print(NB.classes_)
# print(NB.class_prior_)
# print(NB.theta_)
# print(NB.var_)

#print(X_train[L_train == 'Down'].mean())
#print(X_train[L_train == 'Down'].var(ddof=0))
nb_labels = NB.predict(X_test)
print(confusion_table(nb_labels, L_test))
print(NB.predict_proba(X_test)[:5])