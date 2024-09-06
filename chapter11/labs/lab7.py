from matplotlib.pyplot import subplots
import numpy as np
import pandas as pd
from ISLP.models import ModelSpec as MS
from ISLP import load_data
from lifelines import \
     (KaplanMeierFitter,
      CoxPHFitter)
from lifelines.statistics import \
     (logrank_test,
      multivariate_logrank_test)
from ISLP.survival import sim_time
import matplotlib.pyplot as plt

km = KaplanMeierFitter()
coxph = CoxPHFitter

rng = np.random.default_rng(10)
N = 2000
Operators = rng.choice(np.arange(5, 16), N, replace=True)
Center = rng.choice(['A', 'B', 'C'], N)
Time = rng.choice(['Morn.', 'After.', 'Even.'], N, replace=True)
D = pd.DataFrame({ 'Operators': Operators, 'Center': pd.Categorical(Center), 'Time': pd.Categorical(Time)})

model = MS(['Operators', 'Center', 'Time'], intercept=False)
X = model.fit_transform(D)

true_beta = np.array([0.04, -0.3, 0, 0.2, -0.2])
true_linpred = X.dot(true_beta)
hazard = lambda t: 1e-5 * t
cum_hazard = lambda t: 1e-5 * t**2 / 2

W = np.array([sim_time(l, cum_hazard, rng) for l in true_linpred])
D['Wait time'] = np.clip(W, 0, 1000)
D['Failed'] = rng.choice([1, 0], N, p=[0.9, 0.1])

fig, ax = subplots(figsize=(8,8))
by_center = {}
for center, df in D.groupby('Center'):
    by_center[center] = df
    km_center = km.fit(df['Wait time'], df['Failed'])
    km_center.plot(label='Center=%s' % center, ax=ax)
ax.set_title("Probability of Still being on Hold")
plt.show()

fig, ax = subplots(figsize=(8,8))
by_time = {}
for time, df in D.groupby('Time'):
    by_time[time] = df
    km_time = km.fit(df['Wait time'], df['Failed'])
    km_time.plot(label='Time=%s' % time, ax=ax)
ax.set_title("Probability of Still Being on Hold")

plt.show()

# print(multivariate_logrank_test(D['Wait time'],
#                           D['Center'],
#                           D['Failed']))
#
# print(multivariate_logrank_test(D['Wait time'],
#                           D['Time'],
#                           D['Failed']))

X = MS(['Wait time',
        'Failed',
        'Center'],
        intercept=False).fit_transform(D)
F = coxph().fit(X, 'Wait time', 'Failed')
print(F.log_likelihood_ratio_test())

X = MS(['Wait time',
        'Failed',
        'Time'],
       intercept=False).fit_transform(D)
F = coxph().fit(X, 'Wait time', 'Failed')
print(F.log_likelihood_ratio_test())

X = MS(D.columns, intercept=False).fit_transform(D)
fit_queuing = coxph().fit(X, 'Wait time', 'Failed')
print(fit_queuing.summary[['coef', 'se(coef)', 'p']])