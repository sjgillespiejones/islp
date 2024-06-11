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

Weekly = load_data('Weekly')
X = Weekly.loc[:, 'Lag1':'Volume']
y = Weekly.Direction == 'Up'

glm = sm.GLM(y, X, family=sm.families.Binomial())
results = glm.fit()

probs = results.predict()
labels = np.array(['Down'] * Weekly.shape[0])
labels[probs > 0.5] = "Up"

table = confusion_table(labels, Weekly.Direction)
print(table)
print((172 + 427) / (172 + 178 + 312 + 427))

