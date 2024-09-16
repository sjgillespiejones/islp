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

data = pd.DataFrame({
    'time': [26.5, 37.2, 57.3, 90.8, 20.2, 89.8],
    'status': [1, 1, 1, 0, 0, 0],
    'covariate': [0.1, 11, -0.3, 2.8, 1.8, 0.4]
})

data['grouping'] = data['covariate'] >= 2
coxph = CoxPHFitter

group_df = data[['time', 'status', 'grouping']]
model_df = MS(group_df.columns, intercept=False).fit_transform(group_df)
cox_fit = coxph().fit(model_df, 'time', 'status')
print(cox_fit.summary[['coef', 'se(coef)', 'p']])
print(cox_fit.log_likelihood_ratio_test())

by_grouping = {}
for group, df in data.groupby('grouping'):
    by_grouping[group] = df


results = logrank_test(by_grouping[True]['time'], by_grouping[False]['time'], by_grouping[True]['status'], by_grouping[False]['status'])
print(results.summary)