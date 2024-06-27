import itertools

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

Boston = load_data('Boston')
model = MS(Boston.columns.drop('medv'), intercept=False)
D = model.fit_transform(Boston)
feature_names = list(D.columns)
X = np.asarray(D)

(X_train, X_test, y_train, y_test) = skm.train_test_split(X, Boston['medv'], test_size=0.3, random_state=0)

features = np.arange(1, X_train.shape[1], 1)
estimators = np.arange(200, 1_000, 100)

ax = subplots(figsize=(12, 12))[1]
ax.set_xlabel('Number of estimators')
ax.set_ylabel('Mean squared error')

for feature in features:
    mean_squared_errors = np.zeros(len(estimators), dtype=float)
    for index, (estimator) in enumerate(estimators):
        bag_boston = RF(max_features=feature, random_state=0, n_estimators=estimator).fit(X_train, y_train)
        y_hat_bag = bag_boston.predict(X_test)
        squared_error = (y_test - y_hat_bag) ** 2
        mean_squared_errors[index] = np.mean(squared_error)

    label = str(feature) + ' feature(s)'
    ax.plot(estimators, mean_squared_errors, label=label)

ax.legend()
plt.show()