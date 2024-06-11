import matplotlib.pyplot as plt

y1 = 1
lamda = 23

betas = range(-2000,2000)
outputs = [((y1 -b) ** 2) + (lamda * b) for b in betas]

plt.scatter(betas, outputs)

expectedBetaMinimisation = y1 / (1 + lamda)
betaPoint = (y1 - expectedBetaMinimisation) ** 2 + (lamda * expectedBetaMinimisation)
print(expectedBetaMinimisation, betaPoint)
plt.scatter(expectedBetaMinimisation, betaPoint, c='red')
plt.show()