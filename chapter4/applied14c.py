import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,summarize)
import matplotlib.pyplot as plt

from ISLP import confusion_table
from ISLP.models import contrast
from sklearn.discriminant_analysis import \
    (LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

Auto = load_data('Auto')
median_mpg = np.median(Auto['mpg'])
mpg1 = np.where(Auto['mpg'] > median_mpg, 1, 0)
Auto['mpg01'] = mpg1

# Can split on index like so
#train = (Auto.index % 10) < 8

x = Auto[['cylinders', 'displacement', 'weight']].values
y = Auto['mpg01'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
print(x_train)