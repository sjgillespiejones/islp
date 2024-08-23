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
cleaned = BrainCancer.dropna()
all_MS = MS(cleaned.columns, intercept=False)
all_df = all_MS.fit_transform(cleaned)
fit_all = coxph().fit(all_df, 'time', 'status')
# print(fit_all.summary[['coef', 'se(coef)', 'p']])

levels = cleaned['diagnosis'].unique()
def representative(series):
    if(hasattr(series.dtype, 'categories')):
        return pd.Series.mode(series)
    else:
        return series.mean()

modal_Data = cleaned.apply(representative, axis=0)

modal_df = pd.DataFrame([modal_Data.iloc[0] for _ in range(len(levels))])
modal_df['diagnosis'] = levels
# print(modal_df)

modal_X = all_MS.transform(modal_df)
modal_X.index = levels
print(modal_X)

predicted_survival = fit_all.predict_survival_function(modal_X)
print(predicted_survival)

fig, ax = subplots(figsize=(8,8))
predicted_survival.plot(ax=ax)
plt.show()