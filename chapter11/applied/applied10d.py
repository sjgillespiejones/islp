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

BrainCancer = load_data('BrainCancer').dropna()
km = KaplanMeierFitter()

BrainCancer['grouping_property'] = BrainCancer['ki'].apply(lambda x : x if x != 40 else 60)

by_ki = {}
fig, ax = subplots(figsize=(8,8))
for result, df in BrainCancer.groupby('grouping_property'):
    by_ki[result] = df
    km_result = km.fit(df['time'], df['status'])
    km_result.plot(label='ki=%d' % result, ax=ax)

plt.show()

print(BrainCancer.head())