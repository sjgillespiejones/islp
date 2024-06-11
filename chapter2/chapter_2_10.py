import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import pandas as pd

Boston = pd.read_csv('../Boston.csv')

# ng.scatter_matrix(Boston)
# fig, axes = subplots(2, 2, figsize=(8,8))
# Boston.plot.scatter('rad', 'crim', ax=axes[0, 0])
# Boston.plot.scatter('tax', 'crim', ax=axes[0, 1])
# Boston.plot.scatter('lstat', 'crim', ax=axes[1, 0])
# plt.show()

crime_rate_correlation = Boston.corrwith(Boston['crim']).sort_values()

charles_river_suburbs = Boston.query('chas==1')
median_ptratio = np.median(Boston['ptratio'])

lowest_median_value_owner_occupied_homes = Boston.sort_values(by=['medv'])

large_dwelling_suburbs = Boston.query('rm >= 8')
print(Boston.describe())
print(large_dwelling_suburbs.describe())
