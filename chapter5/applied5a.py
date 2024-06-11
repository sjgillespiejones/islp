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

from utils.classification import error_rate

Default = load_data('Default')

Default['default_yes'] = (Default['default'] == 'Yes').astype('int')
x = Default[['income', 'balance']]
y = Default['default_yes']


#f = 'default_yes ~ income + balance'
#res = smf.logit(formula=f, data=Default).fit()
#print(res.summary())

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.95)

logistic_regression = LogisticRegression().fit(x_train, y_train)
probs = logistic_regression.predict(x_test)
logit_table = confusion_table(probs, y_test)
print(logit_table)
print(error_rate(logit_table))