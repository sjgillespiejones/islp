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
km = KaplanMeierFitter()

class TimePoint():
    def __init__(self, Val):
        self.Lower = []
        self.Upper = []
        self.Val = Val
        self.SurvivalRate = []



time_points = { 0: TimePoint(0)}

for i in np.array(BrainCancer['time'].sort_values()):
    time_points[i] = TimePoint(i)

km_curve = km.fit(BrainCancer['time'], BrainCancer['status'])
print(km_curve.survival_function_)

B = 200
for i in range(1, B):
    bootstrapped = bootstrap(BrainCancer, 88)
    km_curve = km.fit(bootstrapped['time'], bootstrapped['status'])
    for (j, k) in km_curve.confidence_interval_.iterrows():
        time_points[j].Lower.append(k['KM_estimate_lower_0.95'])
        time_points[j].Upper.append(k['KM_estimate_upper_0.95'])
    for(j, k) in km_curve.survival_function_.iterrows():
        time_points[j].SurvivalRate.append(k['KM_estimate'])


summary_data = []
for point in time_points:
    time_point = time_points[point]

    summary_data.append({
        'time': time_point.Val,
        'lower_mean': np.mean(time_point.Lower),
        'upper_mean': np.mean(time_point.Upper),
        'survival_se': np.std(time_point.SurvivalRate)
    })

summary_data_frame = pd.DataFrame(summary_data)
print(summary_data_frame.head())