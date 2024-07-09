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

logistic_regression = sm.Logit(y, X).fit()
# print(logistic_regression.summary())

(X_train, X_test, y_train, y_test) = skm.train_test_split(X, y, test_size=0.2, random_state=0)
trained_logit = sklearn.linear_model.LogisticRegression(fit_intercept=True).fit(X_train, y_train)

y_hat_test = trained_logit.predict(X_test)
print(confusion_table(y_hat_test, y_test))

plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_hat_test, cmap=cm.coolwarm)
plt.xlabel('X1')
plt.xlabel('X2')
plt.show()


X['x1_squared'] = X['x1'] ** 2
X['x2_squared'] = X['x1'] **2

(X_train, X_test, y_train, y_test) = skm.train_test_split(X, y, test_size=0.2, random_state=0)
trained_logit = sklearn.linear_model.LogisticRegression(fit_intercept=True).fit(X_train, y_train)

y_hat_test = trained_logit.predict(X_test)
print(confusion_table(y_hat_test, y_test))

plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_hat_test, cmap=cm.coolwarm)
plt.xlabel('X1')
plt.xlabel('X2')
plt.show()

