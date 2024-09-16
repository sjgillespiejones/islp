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

km = KaplanMeierFitter()
fig, ax = subplots(figsize=(8,8))
for result, df in data.groupby('grouping'):
    km_result = km.fit(df['time'], df['status'])
    km_result.plot(label='group=%d' % result, ax=ax)

plt.show()