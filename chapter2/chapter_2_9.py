import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import pandas as pd

Auto = pd.read_csv('../Auto.data', na_values=['?'], delim_whitespace=True).dropna()


fig, axes = subplots(2, 2)
Auto.plot.scatter('displacement', 'mpg', ax=axes[0,0])
Auto.plot.scatter('horsepower', 'mpg', ax=axes[0,1])
Auto.plot.scatter('weight', 'mpg',ax=axes[1,0])
print(Auto.columns)
plt.show()
#quantitative = Auto.select_dtypes(include=['number']).drop(Auto.index[10:85])

# print(quantitative)
# for col in quantitative:
#     data = quantitative[col]
#     range = np.max(data) - np.min(data)
#     print(col, ' Range: ', range)
#     mean = np.mean(data)
#     print('Mean: ', mean)
#     std_dev = np.std(data)
#     print('Standard deviation', std_dev)
