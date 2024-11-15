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

X = USArrests_scaled
U, D, V = np.linalg.svd(X, full_matrices=False)

n_omit = 20
np.random.seed(15)
r_idx = np.random.choice(np.arange(X.shape[0]),
                         n_omit,
                         replace=False)
c_idx = np.random.choice(np.arange(X.shape[1]),
                         n_omit,
                         replace=True)
Xna = X.copy()
Xna[r_idx, c_idx] = np.nan

def low_rank(X, M=1):
    U,D,V = np.linalg.svd(X)
    L = U[:, :M] * D[None, :M]
    return L.dot(V[:M])

Xhat = Xna.copy()
Xbar = np.nanmean(Xhat, axis=0)
Xhat[r_idx, c_idx] = Xbar[c_idx]

thresh = 1e-7
rel_err = 1
count = 0
ismiss = np.isnan(Xna)
mssold = np.mean(Xhat[~ismiss] ** 2)
mss0 = np.mean(Xna[~ismiss] ** 2)

print(mss0)
print(mssold)

while rel_err > thresh:
    count += 1
    Xapp = low_rank(Xhat, M=1)
    Xhat[ismiss] = Xapp[ismiss]
    mss = np.mean(((Xna - Xapp)[~ismiss]) ** 2)
    rel_err = (mssold - mss) / mss0
    mssold = mss
    print("Iteration: {0}, MSS:{1:.3f}, Rel.Err {2:.2e}"
          .format(count, mss, rel_err))

print(np.corrcoef(Xapp[ismiss], X[ismiss])[0,1])