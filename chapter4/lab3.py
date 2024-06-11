import numpy as np
import pandas as pd
import statsmodels.api as sm
from ISLP import confusion_table
from ISLP import load_data
from ISLP.models import (ModelSpec as MS)

Smarket = load_data('Smarket')
model = MS(['Lag1', 'Lag2']).fit(Smarket)
X = model.transform(Smarket)
y = Smarket.Direction == 'Up'

train = (Smarket.Year < 2005)
X_train, X_test = X.loc[train], X.loc[~train]
y_train, y_test = y.loc[train], y.loc[~train]

D = Smarket.Direction
L_train, L_test = D.loc[train], D.loc[~train]

glm_train = sm.GLM(y_train, X_train, family=sm.families.Binomial())
results = glm_train.fit()
probs = results.predict(exog=X_test)
labels = np.array(['Down'] * 252)
labels[probs > 0.5] = 'Up'
table = confusion_table(labels, L_test)

newdata = pd.DataFrame({'Lag1':[1.2, 1.5], 'Lag2': [1.1, -0.8]})
newX = model.transform(newdata)
print(results.predict(newX))