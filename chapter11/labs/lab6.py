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

Publication = load_data('Publication')

posres_df = (MS(['posres', 'time', 'status'], intercept=False)
             .fit_transform(Publication))
posres_fit = CoxPHFitter().fit(posres_df, 'time', 'status')
print(posres_fit.summary[['coef', 'se(coef)', 'p']])

model = MS(Publication.columns.drop('mech'), intercept=False)
print(CoxPHFitter().fit(model.fit_transform(Publication), 'time', 'status')
      .summary[['coef', 'se(coef)', 'p']])