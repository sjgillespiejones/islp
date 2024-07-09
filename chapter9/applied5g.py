import numpy as np
import pandas as pd
import sklearn.linear_model
from matplotlib.pyplot import subplots, cm
import sklearn.model_selection as skm
from ISLP import load_data, confusion_table
from sklearn.svm import SVC
from ISLP.svm import plot as plot_svm
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

rng = np.random.default_rng(5)
x1 = rng.uniform(size=500) - 0.5
x2 = rng.uniform(size=500) - 0.5
y = x1 ** 2 - x2 ** 2 > 0

_, ax = subplots(figsize=(8, 8))
ax.scatter(x1, x2, c=y, cmap=cm.coolwarm)
plt.show()

X = pd.DataFrame({'x1': x1, 'x2':x2})
(X_train, X_test, y_train, y_test) = skm.train_test_split(X, y, test_size=0.2, random_state=0)


radial_svc = SVC(kernel='rbf', gamma=1, C=1)
radial_svc.fit(X_train, y_train)

kfold = skm.KFold(5, random_state=0, shuffle=True)
grid = skm.GridSearchCV(radial_svc, {'C': [0.001, 0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.1, 0.5, 1, 2, 3, 4, 10, 100]}, refit=True, cv=kfold, scoring='accuracy')
grid.fit(X_train, y_train)

best_radial_svm = grid.best_estimator_
_, ax = subplots(figsize=(8, 8))
plot_svm(X_train, y_train, best_radial_svm, ax=ax)
plt.show()

y_hat_test = best_radial_svm.predict(X_test)
print(confusion_table(y_hat_test, y_test))

linear_svc = SVC(kernel='linear')
linear_svc.fit(X_train, y_train)
grid = skm.GridSearchCV(radial_svc, {'C': [0.001, 0.1, 1, 10, 100, 1000]}, refit=True, cv=kfold, scoring='accuracy')
grid.fit(X_train, y_train)

best_linear_svm = grid.best_estimator_
_, ax = subplots(figsize=(8, 8))
plot_svm(X_train, y_train, best_linear_svm, ax=ax)
plt.show()

y_hat_test = best_linear_svm.predict(X_test)
print(confusion_table(y_hat_test, y_test))
