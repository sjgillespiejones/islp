import numpy as np
import statsmodels.api as sm
from ISLP import confusion_table
from ISLP import load_data
from ISLP.models import (ModelSpec as MS)

Smarket = load_data('Smarket')

allvars = Smarket.columns.drop(['Today', 'Direction', 'Year'])
design = MS(allvars)
X = design.fit_transform(Smarket)
y = Smarket.Direction == 'Up'
glm = sm.GLM(y, X, family=sm.families.Binomial())
results = glm.fit()

train = (Smarket.Year < 2005)
Smarket_train = Smarket.loc[train]
Smarket_test = Smarket.loc[~train]

X_train, X_test = X.loc[train], X.loc[~train]
y_train, y_test = y.loc[train], y.loc[~train]
glm_train = sm.GLM(y_train, X_train, family=sm.families.Binomial())
results = glm_train.fit()
probs = results.predict(exog=X_test)
print(probs)

D = Smarket.Direction
L_train, L_test = D.loc[train], D.loc[~train]
labels = np.array(['Down'] * 252)
labels[probs > 0.5] = 'Up'
table = confusion_table(labels, L_test)
print(table)

print(np.mean(labels == L_test), np.mean(labels != L_test))