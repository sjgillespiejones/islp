from ISLP import load_data
import statsmodels.api as sm
import sklearn.model_selection as skm
import numpy as np
from statsmodels.stats.anova import anova_lm

Wage = load_data('Wage')

K = 300
kfold = skm.KFold(K)

MSE = np.zeros(len(Wage.index))

for j, (train_index, test_index) in enumerate(kfold.split(Wage)):
    training_data = Wage.iloc[train_index]
    test_data = Wage.iloc[test_index]
    model = sm.formula.ols(formula='wage ~ age + np.power(age, 2) + np.power(age, 3) + np.power(age, 4) + np.power(age, 5)', data=training_data)
    results = model.fit()
    predictions = results.predict(test_data)
    RSS = np.power(test_data['wage'] - predictions, 2)
    MSE[j] = np.mean(RSS)

print(np.mean(MSE))


