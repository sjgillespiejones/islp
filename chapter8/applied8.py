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
print(Carseats.head())

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

reg = DTR(max_depth=3)
reg.fit(X_train, y_train)
ax = subplots(figsize=(12,12))[1]
plot_tree(reg, feature_names=feature_names, ax=ax)
plt.show()

depth3_tree = (y_test - reg.predict(X_test)) ** 2
print(np.mean(depth3_tree))

ccp_path = reg.cost_complexity_pruning_path(X_train, y_train)
kfold = skm.KFold(5)
grid = skm.GridSearchCV(reg, { 'ccp_alpha': ccp_path.ccp_alphas}, refit=True, cv=kfold, scoring='neg_mean_squared_error')
fitted_grid = grid.fit(X_train, y_train)
best_ = grid.best_estimator_
best_estimator_error = ((y_test - best_.predict(X_test)) ** 2)
print(np.mean(best_estimator_error))

ax = subplots(figsize=(12,12))[1]
plot_tree(fitted_grid.best_estimator_, feature_names=feature_names, ax=ax)
plt.show()
