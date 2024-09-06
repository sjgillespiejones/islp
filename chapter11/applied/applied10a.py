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

BrainCancer = load_data('BrainCancer').dropna()

fig, ax = subplots(figsize=(8,8))
km_curve = km.fit(BrainCancer['time'], BrainCancer['status'])
km_curve.plot(label='Kaplan Meier fit', ax=ax)
plt.grid()
plt.show()