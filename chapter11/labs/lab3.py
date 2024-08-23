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

BrainCancer = load_data('BrainCancer')

coxph = CoxPHFitter
sex_df = BrainCancer[['time', 'status', 'sex']]
model_df = MS(['time', 'status', 'sex'], intercept=False).fit_transform(sex_df)
cox_fit = coxph().fit(model_df, 'time', 'status')
print(cox_fit.summary[['coef', 'se(coef)', 'p']])
print(cox_fit.log_likelihood_ratio_test())