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

clf = DTC(criterion='entropy', max_depth=3, random_state=0)
clf.fit(X, High)

#print(accuracy_score(High, clf.predict(X)))
resid_dev = np.sum(log_loss(High, clf.predict_proba(X)))
#print(resid_dev)

ax = subplots(figsize=(12,12))[1]
plot_tree(clf, feature_names=feature_names, ax=ax)
plt.show()

text = export_text(clf, feature_names=feature_names, show_weights=True)
#print(text)

validation = skm.ShuffleSplit(n_splits=1, test_size=200, random_state=0)
results = skm.cross_validate(clf, D, High, cv=validation)
# correct predictions for 68% of locations in the test set
print(results['test_score'])