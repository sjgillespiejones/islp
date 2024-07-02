import numpy as np
from matplotlib.pyplot import subplots, cm
import sklearn.model_selection as skm
from ISLP import load_data, confusion_table
from sklearn.svm import SVC
from ISLP.svm import plot as plot_svm
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

roc_curve = RocCurveDisplay.from_estimator

Khan = load_data('Khan')
X_train = Khan['xtrain']
y_train = Khan['ytrain']
X_test = Khan['xtest']
y_test = Khan['ytest']
print(y_test)
print(X_train.shape, X_test.shape)

khan_linear = SVC(kernel='linear', C=10)
khan_linear.fit(X_train, y_train)

print(confusion_table(khan_linear.predict(X_test), y_test))