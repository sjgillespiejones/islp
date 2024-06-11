import matplotlib.pyplot as plt
n = 500
prob = [1 - (1 - 1/n)**n for n in range(1, n)]
plt.scatter(range(1, n), prob)
plt.xlim(0, n)
plt.ylim(0.6, 0.7)
plt.show()