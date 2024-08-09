import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from matplotlib.pyplot import subplots

from utils.summaryPlot import summary_plot

lstm_results = pd.read_csv('logs/IMDB_LSTM/version_0/metrics.csv')
fig, ax = subplots(1, 1, figsize=(6, 6))
summary_plot(lstm_results,
             ax,
             col='accuracy',
             ylabel='Accuracy')
ax.set_xticks(np.linspace(0, 20, 5).astype(int))
ax.set_ylabel('Accuracy')
ax.set_ylim([0.5, 1])
plt.show()