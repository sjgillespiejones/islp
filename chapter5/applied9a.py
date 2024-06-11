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

Boston = load_data('Boston')
mu = np.mean(Boston['medv'])
n = len(Boston)
mu_standard_error_estimate = Boston['medv'].std() / np.sqrt(n)

def bootstrap(dataframe, sample_size=None):
    if(sample_size == None):
        sample_size = len(dataframe)

    bootSample_i = (np.random.rand(sample_size) * len(dataframe)).astype(int)
    bootSample_i = np.array(bootSample_i)
    bootSample_dataframe = dataframe.iloc[bootSample_i]
    return bootSample_dataframe

boostrapped_medv = bootstrap(Boston['medv'], 10000)
boostrapped_se_estimate = boostrapped_medv.std() / np.sqrt(n)
print(mu_standard_error_estimate)
print(boostrapped_se_estimate)

bootstrapped_mean = boostrapped_medv.mean()
bootstrapped_conf = [bootstrapped_mean - (2 * boostrapped_se_estimate), mu + (2 * boostrapped_se_estimate)]
print(bootstrapped_conf)

print(Boston['medv'].median())
medians = [Boston['medv'].sample(n = len(Boston), replace=True).median() for _ in range(10000)]
print(np.std(medians))

tenth_percentile = np.percentile(Boston['medv'], 10)
print(tenth_percentile)

tenth_percentiles = [np.percentile(Boston['medv'].sample(n = len(Boston), replace=True), 10) for _ in range(10000)]
print(np.std(tenth_percentiles))

