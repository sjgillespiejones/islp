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

np.random.seed(0)
X = np.random.standard_normal((50,2))
X[:25, 0] += 3
X[:25, 1] -= 4

kmeans = KMeans(n_clusters=3, random_state=2, n_init=20).fit(X)
print(kmeans.labels_)

fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.scatter(X[:,0], X[:,1], c=kmeans.labels_)
ax.set_title("K-Means Clustering Results with K=3");
plt.show()

kmeans1 = KMeans(n_clusters=3, random_state=3, n_init=1).fit(X)
kmeans20 = KMeans(n_clusters=3, random_state=3, n_init=20).fit(X)
print(kmeans1.inertia_)
print(kmeans20.inertia_)