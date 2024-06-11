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

knn1 = KNeighborsClassifier(n_neighbors=3)
knn1.fit(X_train, L_train)
knn1_pred = knn1.predict(X_test)
print(confusion_table(knn1_pred, L_test))
#print((43 + 83) / 252, np.mean(knn1 == L_test))