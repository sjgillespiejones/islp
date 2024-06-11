import numpy as np
import random

rng = np.random.default_rng(10)
store = np.empty(10000)
for i in range(10000):
    store[i] = np.sum(rng.choice(100, replace=True) == 4) > 0
print(store)

n = 100
N = 10000
count = 0
for _ in range(N):
    count += 4 in random.choices(range(1, n + 1), k=n)

print(count/N)