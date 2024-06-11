import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots

import pandas as pd

Auto = pd.read_csv('Auto.data', na_values=['?'], delim_whitespace=True).dropna()
# fig, axes = subplots(ncols=3, figsize=(15,5))
# Auto.plot.scatter('horsepower', 'mpg', ax=axes[1])
#
# plt.show()
print(Auto['cylinders'].describe())
print(Auto['mpg'].describe())
