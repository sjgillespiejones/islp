import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,summarize)
import matplotlib.pyplot as plt
import seaborn as sns

from ISLP import confusion_table
from ISLP.models import contrast
from sklearn.discriminant_analysis import \
    (LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from patsy.highlevel import dmatrices

from chapter4.applied14f import success_rate

Boston = load_data('Boston')
median_crim = np.median(Boston['crim'])
crim01 = np.where(Boston['crim'] > median_crim, 1, 0)
Boston['crim01'] = crim01

# g = sns.PairGrid(Boston)
# g.map_upper(plt.scatter, s=3)
# g.map_diag(plt.hist)
# g.map_lower(sns.kdeplot, cmap="Blues_d")
# g.fig.set_size_inches(12, 12)

#sns.pairplot(Boston)
#plt.show()

#print(Boston.corr())

# y, X = dmatrices('crim01 ~ nox + rad + tax', data=Boston, return_type='dataframe')
# logit = sm.Logit(y, X)
# result = logit.fit()
# print(result.summary())

x = Boston[['nox', 'rad', 'tax']].values
y = Boston['crim01'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

lda = LDA(store_covariance=True)
lda.fit(x_train, y_train)
probs = lda.predict(x_test)
lda_table = confusion_table(probs, y_test)
print("--LDA--")
print(lda_table)
print(success_rate(lda_table))

qda = QDA(store_covariance=True)
qda.fit(x_train, y_train)
probs = qda.predict(x_test)
qda_table = confusion_table(probs, y_test)
print("\n--QDA--")
print(qda_table)
print(success_rate(qda_table))

logit = LogisticRegression().fit(x_train, y_train)
probs = logit.predict(x_test)
logit_table = confusion_table(probs, y_test)
print("\n--Logistic regression--")
print(logit_table)
print(success_rate(logit_table))

NB = GaussianNB()
NB.fit(x_train, y_train)
nb_labels = NB.predict(x_test)
naive_bayes_table = confusion_table(nb_labels, y_test)
print("\n--Naive bayes--")
print(naive_bayes_table)
print(success_rate(naive_bayes_table))

print("\n--K Nearest neighbours--")
for K in range(1, 20) :
    knn = KNeighborsClassifier(n_neighbors=K)
    knn_pred = knn.fit(x_train, y_train).predict(x_test)
    table = confusion_table(knn_pred, y_test)
    print('K = ' + str(K))
    print(table)
    print(success_rate(table))
    print('\n')