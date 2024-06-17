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

bart_boston = BART(random_state=0, burnin=5, ndraw=15)
bart_boston.fit(X_train, y_train)

yhat_test = bart_boston.predict(X_test.astype(np.float32))
squared_error = (y_test - yhat_test) ** 2
print(np.mean(squared_error))

var_inclusion = pd.Series(bart_boston.variable_inclusion_.mean(0), index=D.columns)
print(var_inclusion)