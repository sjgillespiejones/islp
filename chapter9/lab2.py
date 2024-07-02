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
X = rng.standard_normal((200, 2))
X[:100] += 2
X[100:150] -= 2
y = np.array([1] * 150 + [2] * 50)

_, ax = subplots(figsize=(8, 8))
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm.coolwarm)
plt.show()

(X_train, X_test, y_train, y_test) = skm.train_test_split(X, y, test_size=0.5, random_state=0)
# radial kernel
svm_rbf = SVC(kernel='rbf', gamma=1, C=1)
svm_rbf.fit(X_train, y_train)

_, ax = subplots(figsize=(8, 8))
plot_svm(X_train, y_train, svm_rbf, ax=ax)
plt.show()

kfold = skm.KFold(5, random_state=0, shuffle=True)
grid = skm.GridSearchCV(svm_rbf, {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.5, 1, 2, 3, 4]}, refit=True, cv=kfold,
                        scoring='accuracy')
grid.fit(X_train, y_train)
# print(grid.best_params_)

best_svm = grid.best_estimator_
_, ax = subplots(figsize=(8, 8))
plot_svm(X_train, y_train, best_svm, ax=ax)
plt.show()

y_hat_test = best_svm.predict(X_test)
print(confusion_table(y_hat_test, y_test))

_, ax = subplots(figsize=(8, 8))
roc_curve(best_svm, X_train, y_train, name='Training', color='r', ax=ax)
plt.show()

svm_flex = SVC(kernel='rbf', gamma=50, C=1)
svm_flex.fit(X_train, y_train)
_, ax = subplots(figsize=(8, 8))
roc_curve(svm_flex, X_test, y_test, name='Training $\gamma=50$', color='r', ax=ax)
plt.show()

fig, ax = subplots(figsize=(8, 8))
for (X_, y_, c, name) in zip((X_train, X_test), (y_train, y_test), ('r', 'b'), ('CV tuned on training', 'CV tuned on test')):
    roc_curve(best_svm, X_, y_, name=name, ax=ax, color=c)
    plt.show()
