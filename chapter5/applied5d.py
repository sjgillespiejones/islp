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
Default['student_yes'] = (Default['student'] == 'Yes').astype('int')

x = Default[['income', 'balance', 'student_yes']]
y = Default['default_yes']

f = 'default_yes ~ income + balance + student_yes'
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.9)
train = x_train.join(y_train)
res = smf.logit(formula=f, data=train).fit()
#print(res.summary())

y_pred = (res.predict(x_test) > 0.5) * 1
table = confusion_table(y_pred, y_test)
print(table)
print(error_rate(table))