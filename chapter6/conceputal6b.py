import matplotlib.pyplot as plt
import numpy as np

y1 = 0
lamda = 4

betas = range(-2000,2000)
outputs = [((y1 -b) ** 2) + (lamda * np.abs(b)) for b in betas]

def expectedMinimisation(y1, lamda):
    if(y1 > lamda / 2):
        return y1 - (lamda / 2)
    if(y1 < -lamda / 2):
        return y1 + lamda / 2
    return 0

expectedBetaMinimisation = expectedMinimisation(y1, lamda)
betaPoint = (y1 - expectedBetaMinimisation) ** 2 + (lamda * np.abs(expectedBetaMinimisation))

plt.scatter(betas, outputs)
plt.scatter(expectedBetaMinimisation, betaPoint, c='red')
plt.show()
