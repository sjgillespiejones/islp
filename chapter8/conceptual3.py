import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots

def classification_error(prob1, prob2):
    return 1 - np.max([prob1, prob2])

def gini_index(prob1, prob2):
    return (prob1 * (1 - prob1)) + (prob2 * (1 - prob2))

def entropy(prob1, prob2):
    return -1 * (( prob1 * np.log(prob1)) + (prob2 * np.log(prob2)))

prob1s = np.linspace(0, 1)
prob2s = 1 - prob1s

classification_errors = np.zeros(prob1s.shape)
gini_indices = np.zeros(prob1s.shape)
entropies = np.zeros(prob1s.shape)

for index, (p1, p2) in enumerate(zip(prob1s, prob2s)):
    classification_errors[index] = classification_error(p1, p2)
    gini_indices[index] = gini_index(p1, p2)
    entropies[index] = entropy(p1, p2)

ax = subplots(figsize=(8,8))[1]
ax.plot(prob1s, classification_errors, 'r', label='Classification Error')
ax.plot(prob1s, gini_indices, 'g', label='Gini index')
ax.plot(prob1s, entropies, 'y', label='Entropy')
ax.set_xlabel('Prob(class1)')
ax.set_ylabel('Prob(class2)')
ax.legend()
plt.show()