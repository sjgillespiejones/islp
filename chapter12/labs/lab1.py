import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.datasets import get_rdataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ISLP import load_data

from sklearn.cluster import \
     (KMeans,
      AgglomerativeClustering)
from scipy.cluster.hierarchy import \
     (dendrogram,
      cut_tree)
from ISLP.cluster import compute_linkage

USArrests = get_rdataset('USArrests').data

scaler = StandardScaler(with_mean=True, with_std=True)
USArrests_scaled = scaler.fit_transform(USArrests)

pcaUS = PCA()
pcaUS.fit(USArrests_scaled)
scores = pcaUS.transform(USArrests_scaled)
i, j = 0, 1
fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.scatter(scores[:, 0], scores[:, 1])
ax.set_xlabel(f'PC{i+1}')
ax.set_ylabel(f'PC{j+1}')
for k in range(pcaUS.components_.shape[1]):
    ax.arrow(0, 0, pcaUS.components_[i,k], pcaUS.components_[j,k])
    ax.text(pcaUS.components_[i,k], pcaUS.components_[j,k], USArrests.columns[k])

scale_arrow = s_ = 2
scores[:, 1] *= -1
pcaUS.components_[1] *= -1
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(scores[:,0], scores[:,1])
ax.set_xlabel('PC%d' % (i+1))
ax.set_ylabel('PC%d' % (j+1))
for k in range(pcaUS.components_.shape[1]):
    ax.arrow(0, 0, s_*pcaUS.components_[i,k], s_*pcaUS.components_[j,k])
    ax.text(s_*pcaUS.components_[i,k],
            s_*pcaUS.components_[j,k],
            USArrests.columns[k])
plt.show()

print(scores.std(0, ddof=1))
print(pcaUS.explained_variance_ratio_)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ticks = np.arange(pcaUS.n_components_) + 1
ax = axes[0]
ax.plot(ticks, pcaUS.explained_variance_ratio_, marker='o')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Proportion of Variance Explained')
ax.set_ylim([0,1])
ax.set_xticks(ticks)

ax = axes[1]
ax.plot(ticks,
        pcaUS.explained_variance_ratio_.cumsum(),
        marker='o')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Cumulative Proportion of Variance Explained')
ax.set_ylim([0, 1])
ax.set_xticks(ticks)
plt.show()

