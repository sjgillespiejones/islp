import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
from statsmodels.datasets import get_rdataset
import sklearn.model_selection as skm
from ISLP import load_data , confusion_table
from ISLP.models import ModelSpec as MS
from sklearn.tree import (DecisionTreeClassifier as DTC, DecisionTreeRegressor as DTR,plot_tree ,export_text)
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
y = Carseats['Sales']

feature_names = list(X.columns)
#print(X.dtypes)

(X_train, X_test, y_train, y_test) = skm.train_test_split(X, y, test_size=0.3, random_state=0)

random_forests_carseats = RF(max_features=5).fit(X_train, y_train)
y_hat_RF = random_forests_carseats.predict(X_test)
squared_error = (y_test - y_hat_RF) ** 2
print(np.mean(squared_error))

feature_imp = pd.DataFrame({ 'importance': random_forests_carseats.feature_importances_}, index=feature_names)
print(feature_imp.sort_values(by='importance', ascending=False))

