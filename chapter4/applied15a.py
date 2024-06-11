import matplotlib.pyplot as plt
import numpy as np


def Power():
    return 2 ** 3

#print(Power())

def Power2(x, a):
    return x ** a

#print(Power2(3, 8))
#print(Power2(10, 3))
#print(Power2(8, 17))
#print(Power2(131, 3))

def PlotPower(array, power):
    y = Power2(array, power)
    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('y = x^' + str(power))
    ax.set_title('Power2()')
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.plot(array, y)
    plt.show()

PlotPower(np.arange(1, 11), 3)