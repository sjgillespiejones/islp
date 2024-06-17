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

Carseats = load_data('Carseats')
High = np.where(Carseats.Sales > 8, "Yes", "No")

model = MS(Carseats.columns.drop('Sales'), intercept=False)

D = model.fit_transform(Carseats)
feature_names = list(D.columns)

X = np.asarray(D)

(X_train, X_test, High_train, High_test) = skm.train_test_split(X, High, test_size=0.5, random_state=0)

clf = DTC(criterion='entropy', random_state=0)
clf.fit(X_train, High_train)
print(accuracy_score(High_test, clf.predict(X_test)))

ccp_path = clf.cost_complexity_pruning_path(X_train, High_train)
kfold = skm.KFold(10, random_state=1, shuffle=True)
grid = skm.GridSearchCV(clf, { 'ccp_alpha': ccp_path.ccp_alphas}, refit=True, cv=kfold, scoring='accuracy')
grid.fit(X_train, High_train)
print(grid.best_score_)

ax = subplots(figsize=(12,12))[1]
best_ = grid.best_estimator_
plot_tree(best_, feature_names=feature_names, ax=ax)
plt.show()
print(accuracy_score(High_test, best_.predict(X_test)))
confusion = confusion_table(best_.predict(X_test), High_test)
print(confusion)