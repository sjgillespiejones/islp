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

Weekly = load_data('Weekly')
X = Weekly.loc[:, 'Lag2']
X_reshaped = X.values.reshape(-1, 1)
y = Weekly.Direction == 'Up'

train = Weekly.Year < 2009
X_train, X_test = X.loc[train], X.loc[~train]
y_train, y_test = y.loc[train], y.loc[~train]

X_train_reshaped = X_train.values.reshape(-1, 1)
X_test_reshaped = X_test.values.reshape(-1, 1)

Direction = Weekly.Direction
L_train, L_test = Direction.loc[train], Direction.loc[~train]

NB = GaussianNB()
NB.fit(X_train_reshaped, L_train)
nb_labels = NB.predict(X_test_reshaped)
print(confusion_table(nb_labels, L_test))