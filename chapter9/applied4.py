import numpy as np
from matplotlib.pyplot import subplots, cm
import sklearn.model_selection as skm
from ISLP import load_data, confusion_table
from sklearn.svm import SVC
from ISLP.svm import plot as plot_svm
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

rng = np.random.default_rng(100)

X = rng.standard_normal((200, 2))
X[:100] += 2
X[100:150] -= 2
y = np.array([1] * 150 + [2] * 50)

_, ax = subplots(figsize=(8, 8))
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm.coolwarm)
plt.show()

(X_train, X_test, y_train, y_test) = skm.train_test_split(X, y, test_size=0.2, random_state=0)

radial_svc = SVC(kernel='rbf', gamma=1, C=1)
radial_svc.fit(X_train, y_train)

_, ax = subplots(figsize=(8, 8))
plot_svm(X_train, y_train, radial_svc, ax=ax)
plt.show()

kfold = skm.KFold(5, random_state=0, shuffle=True)
grid = skm.GridSearchCV(radial_svc, {'C': [0.001, 0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.1, 0.5, 1, 2, 3, 4, 10, 100]}, refit=True, cv=kfold, scoring='accuracy')
grid.fit(X_train, y_train)

best_radial_svm = grid.best_estimator_
_, ax = subplots(figsize=(8, 8))
plot_svm(X_train, y_train, best_radial_svm, ax=ax)
plt.show()

y_hat_test = best_radial_svm.predict(X_test)
print(confusion_table(y_hat_test, y_test))

poly_svc = SVC(kernel='poly', degree=2, C=1)
poly_svc.fit(X_train, y_train)
_, ax = subplots(figsize=(8, 8))
plot_svm(X_train, y_train, poly_svc, ax=ax)
plt.show()

# grid = skm.GridSearchCV(poly_svc, {'C': [0.001, 0.1, 1, 10, 100, 1000], 'degree': [2, 3, 4, 5]}, refit=True, cv=kfold, scoring='accuracy')
# grid.fit(X_train, y_train)
# print(grid.best_params_)

# best_params are C: 10, 'degree': 4

best_poly_svc = SVC(kernel='poly', degree=4, C=10)
best_poly_svc.fit(X_train, y_train)
_, ax = subplots(figsize=(8, 8))
plot_svm(X_train, y_train, best_poly_svc, ax=ax)
plt.show()

y_hat_test = best_poly_svc.predict(X_test)
print(confusion_table(y_hat_test, y_test))