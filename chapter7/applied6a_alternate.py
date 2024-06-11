import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import f_regression
from ISLP import load_data

Wage = load_data('Wage')

y = Wage['wage'][:, np.newaxis]
X = Wage['age'][:, np.newaxis]

print(X)
scores = []
for i in range(0, 11):
    model = Pipeline([('poly', PolynomialFeatures(degree=i)), ('linear', LinearRegression())])
    model.fit(X, y)
    score = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
    scores.append(np.mean(score))

scores = np.abs(scores)

x_plot = np.arange(0, 11)

plt.plot(x_plot, scores)
plt.ylabel('Mean squared error (CV)')
plt.xlabel('Degrees')
plt.xlim(0, 10)
plt.show()

models = []
for i in range(0, 11):
    poly = PolynomialFeatures(degree=i)
    X_pol = poly.fit_transform(X)
    model = sm.GLS(y, X_pol).fit()
    models.append(model)
print(sm.stats.anova_lm(*models, type=1))

optimal_degree = 4
model = Pipeline([('poly', PolynomialFeatures(degree = optimal_degree)), ('linear', LinearRegression())])
model.fit(X,y)

X_lin = np.linspace(18,80)[:,np.newaxis]
y_lin = model.predict(X_lin)

plt.scatter(X,y)
plt.plot(X_lin, y_lin,'-r')
plt.show()