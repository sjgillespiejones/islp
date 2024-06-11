import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from ISLP import confusion_table
from ISLP import load_data
from ISLP.models import (ModelSpec as MS)

Smarket = load_data('Smarket')
print(Smarket.columns)
print(Smarket.corr())
Smarket.plot(y ='Volume')
plt.show()

allvars = Smarket.columns.drop(['Today', 'Direction', 'Year'])
design = MS(allvars)
X = design.fit_transform(Smarket)
y = Smarket.Direction == 'Up'
glm = sm.GLM(y, X, family=sm.families.Binomial())
results = glm.fit()
print(results.summary())
print(results.params)
print(results.pvalues)
probs = results.predict()
print(probs[:10])
labels = np.array(['Down'] * 1250)
labels[probs > 0.5] = "Up"
print(labels)
table = confusion_table(labels, Smarket.Direction)
print(table)

print((507 + 145) / 1250, np.mean(labels == Smarket.Direction))