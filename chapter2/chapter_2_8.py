import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import pandas as pd

college = (pd.read_csv('../College.csv')
           .rename({'Unnamed: 0': 'College'}, axis=1)
           .set_index('College'))
fig, axes = subplots(2, 2)
college.hist('Room.Board', bins=10, ax=axes[0, 0])
college.hist('Apps', bins=20, ax=axes[0, 1])
college.hist('Grad.Rate', bins=10, ax=axes[1, 0])
college.hist('S.F.Ratio', bins=10, ax=axes[1,1])
plt.show()
