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

from utils.bootstrap import bootstrap

BrainCancer = load_data('BrainCancer').dropna()
coxph = CoxPHFitter


all_MS = MS(BrainCancer.columns, intercept=False)
all_df = all_MS.fit_transform(BrainCancer)
cox_fit = coxph().fit(all_df, 'time', 'status')
print(cox_fit.summary[['coef', 'se(coef)', 'p']])

# summary
# All diagnosis types have low p values, seem important to the model.
# Ki values seem very important as well