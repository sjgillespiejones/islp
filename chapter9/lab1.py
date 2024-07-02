import numpy as np
from matplotlib.pyplot import subplots, cm
import sklearn.model_selection as skm
from ISLP import load_data, confusion_table
from sklearn.svm import SVC
from ISLP.svm import plot as plot_svm
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

roc_curve = RocCurveDisplay.from_estimator

rng = np.random.default_rng(1)
X = rng.standard_normal((50, 2))
y = np.array([-1] * 25 + [1] * 25)
X[y == 1] += 1

fig, ax = subplots(figsize=(8, 8))
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm.coolwarm)
plt.show()

svm_linear = SVC(C=10, kernel='linear')
svm_linear.fit(X, y)

fig, ax = subplots(figsize=(8, 8))
plot_svm(X, y, svm_linear, ax=ax)
plt.show()

svm_linear_small = SVC(C=0.1, kernel='linear')
svm_linear_small.fit(X, y)
fig, ax = subplots(figsize=(8, 8))
plot_svm(X, y, svm_linear_small, ax=ax)
plt.show()

print(svm_linear.coef_)

kfold = skm.KFold(random_state=0, shuffle=True)
grid = skm.GridSearchCV(svm_linear, {'C': [0.001, 0.01, 0.1, 1, 5, 10, 100]}, refit=True, cv=kfold, scoring='accuracy')
grid.fit(X, y)
print(grid.best_params_)
# print(grid.cv_results_)
print(grid.cv_results_[('mean_test_score')])

X_test = rng.standard_normal((20, 2))
y_test = np.array([-1] * 10 + [1] * 10)
X_test[y_test == 1] += 1

best_ = grid.best_estimator_
y_test_hat = best_.predict(X_test)
print(confusion_table(y_test_hat, y_test))

svm_ = SVC(C=0.001, kernel='linear').fit(X, y)
y_test_hat = svm_.predict(X_test)
print(confusion_table(y_test_hat, y_test))

X[y == 1] += 1
fig, ax = subplots(figsize=(8,8))
ax.scatter(X[:,0], X[:, 1], c=y, cmap=cm.coolwarm)
plt.show()

svm_ = SVC(C=1e5, kernel='linear').fit(X, y)
y_hat = svm_.predict(X)
print(confusion_table(y_hat, y))

fig, ax = subplots(figsize=(8,8))
plot_svm(X, y, svm_, ax=ax)
plt.show()

svm_ = SVC(C=0.1, kernel='linear').fit(X, y)
y_hat = svm_.predict(X)
print(confusion_table(y_hat, y))

fig, ax = subplots(figsize=(8,8))
plot_svm(X, y, svm_, ax=ax)
plt.show()