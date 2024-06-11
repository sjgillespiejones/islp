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
model = MS(['Lag2']).fit(Weekly)
X = model.transform(Weekly)
y = Weekly.Direction == 'Up'

train = Weekly.Year < 2009
X_train, X_test = X.loc[train], X.loc[~train]
y_train, y_test = y.loc[train], y.loc[~train]

Direction = Weekly.Direction
L_train, L_test = Direction.loc[train], Direction.loc[~train]

lda = LDA(store_covariance=True)
X_train, X_test = [M.drop(columns=['intercept']) for M in [X_train, X_test]]
lda.fit(X_train, L_train)

probs = lda.predict(X_test)
#labels = np.array(['Down'] * X_test.shape[0])
#labels[probs > 0.5] = "Up"

table = confusion_table(probs, L_test)
print(table)
print((9 + 56) / (9 + 56 + 5 + 34))
