import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from matplotlib.pyplot import subplots

from utils.summaryPlot import summary_plot

cifar_results = pd.read_csv('logs/CIFAR100/version_0/metrics.csv')
fig, ax = subplots(1, 1, figsize=(6, 6))
summary_plot(cifar_results,
             ax,
             col='accuracy',
             ylabel='Accuracy')
ax.set_xticks(np.linspace(0, 30, 6).astype(int))
ax.set_ylabel('Accuracy')
ax.set_ylim([0, 1]);
plt.show()