import numpy as np
from ISLP import load_data
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

College = load_data('College').dropna()
print(College.columns)
# College.rename(columns={"F.Undergrad": "FUndergrad", "P.Undergrad": "PUndergrad", "Room.Board": "RoomBoard",
#                         "S.F.Ratio": "SFRatio", "perc.alumni": "percAlumni", "Grad.Rate": "GradRate"}, errors="raise",
#                inplace=True)
# print(College.columns)
Y = College['Apps']
X = College.drop(['Apps', 'Private'], axis=1)

# train, test = train_test_split(College, random_state=1)
#
# model = sm.formula.ols(
#     formula='Apps ~ Private + Accept + Enroll + Top10perc + Top25perc + FUndergrad + PUndergrad + Outstate + RoomBoard + Books + Personal + PhD + Terminal + SFRatio + percAlumni + Expend + GradRate',
#     data=College)
# results = model.fit()
# print(results.summary())

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
lr = LinearRegression()
lr.fit(X_train, y_train)

mse = np.mean((lr.predict(X_test) - y_test) ** 2)
print(mse)
score = lr.score(X_test, y_test)
# R^2
print(score)
