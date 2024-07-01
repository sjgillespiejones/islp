import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
from statsmodels.datasets import get_rdataset
import sklearn.model_selection as skm
from ISLP import load_data , confusion_table
from ISLP.models import ModelSpec as MS
from sklearn.tree import (DecisionTreeClassifier as DTC, DecisionTreeRegressor as DTR,plot_tree ,export_text)
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import (accuracy_score ,log_loss)
from sklearn.ensemble import(RandomForestRegressor as RF, GradientBoostingRegressor as GBR)
from ISLP.bart import BART
import matplotlib.pyplot as plt

Carseats = load_data('Carseats').dropna()

X = Carseats.drop(['Sales'], axis=1)
shelvelocDummies = pd.get_dummies(X['ShelveLoc'])
X['Shelveloc_Bad'] = shelvelocDummies['Bad']
X['Shelveloc_Medium'] = shelvelocDummies['Medium']
X['Shelveloc_Good'] = shelvelocDummies['Good']
X['Urban_Yes'] = (X['Urban'] == 'Yes').astype(int)
X['US_Yes'] = (X['US'] == 'Yes').astype(int)
X = X.drop(columns=['ShelveLoc', 'Urban', 'US'])
X = np.asarray(X)
y = Carseats['Sales']

#print(X.dtypes)

(X_train, X_test, y_train, y_test) = skm.train_test_split(X, y, test_size=0.3, random_state=0)

bart_carseats = BART(random_state=0, burnin=5, ndraw=15)
bart_carseats.fit(X_train, y_train)

yhat_test = bart_carseats.predict(X_test.astype(np.float32))
squared_error = (y_test - yhat_test) ** 2
print(np.mean(squared_error))