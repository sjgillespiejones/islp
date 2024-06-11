import numpy as np
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

x = rng.normal(size=100)
epsilon = rng.normal(size=100)
betaZero = 5
betaOne = 2
betaTwo = 3.5
betaThree = -2.22

y = betaZero + (betaOne * x) + (betaTwo * (x ** 2)) + (betaThree * (x ** 3)) + epsilon

_, ax = subplots(figsize=(8,8))
ax.scatter(x, y)
plt.show()
