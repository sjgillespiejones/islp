import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,summarize)
import matplotlib.pyplot as plt
import seaborn as sns

from ISLP import confusion_table
from ISLP.models import contrast
from sklearn.discriminant_analysis import \
    (LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from patsy.highlevel import dmatrices
import statsmodels.formula.api as smf

rng = np.random.default_rng(1)
x = rng.normal(size=100)
y = x - 2 * x **2 + rng.normal(size=100)

_, ax = subplots(figsize=(8, 8))
ax.scatter(x, y)
plt.scatter(x, y)
plt.show()