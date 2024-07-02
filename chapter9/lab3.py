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
y = np.array([1] * 150 + [2] * 50)

rng = np.random.default_rng(123)
X = np.vstack([X, rng.standard_normal((50, 2))])
y = np.hstack([y, [0] * 50])
X[y==0,1] += 2

_, ax = subplots(figsize=(8, 8))
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm.coolwarm)
plt.show()

svm_rbf_3 = SVC(kernel='rbf', C=10, gamma=1, decision_function_shape='ovo')
svm_rbf_3.fit(X, y)

_, ax = subplots(figsize=(8,8))
plot_svm(X, y, svm_rbf_3, scatter_cmap=cm.tab10, ax=ax)
plt.show()