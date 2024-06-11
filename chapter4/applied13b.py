import statsmodels.api as sm
from ISLP import load_data

Weekly = load_data('Weekly')
X = Weekly.loc[:, 'Lag1':'Volume']
y = Weekly.Direction == 'Up'
sm_logit = sm.Logit(y, sm.add_constant(X)).fit()
print(sm_logit.summary())
