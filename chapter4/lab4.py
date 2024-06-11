import numpy as np
from ISLP import load_data
from ISLP.models import (ModelSpec as MS)

from sklearn.discriminant_analysis import \
    (LinearDiscriminantAnalysis as LDA)

Smarket = load_data('Smarket')

model = MS(['Lag1', 'Lag2']).fit(Smarket)
X = model.transform(Smarket)
y = Smarket.Direction == 'Up'

train = (Smarket.Year < 2005)
X_train, X_test = X.loc[train], X.loc[~train]
y_train, y_test = y.loc[train], y.loc[~train]
D = Smarket.Direction
L_train, L_test = D.loc[train], D.loc[~train]

lda = LDA(store_covariance=True)
X_train, X_test = [M.drop(columns=['intercept']) for M in [X_train, X_test]]
lda.fit(X_train, L_train)

lda_pred = lda.predict(X_test)

#print(confusion_table(lda_pred, L_test))
lda_prob = lda.predict_proba(X_test)
print(np.all(np.where(lda_prob[:, 1] >= 0.5, 'Up', 'Down') == lda_pred))

print(np.all(
    [lda.classes_[i] for i in np.argmax(lda_prob, 1)] == lda_pred
))

print(np.sum(lda_prob[:, 0] > 0.9))