import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

def Gauss():
    r1 = random.random()
    r2 = random.random()
    return np.sqrt(-2*np.log(r1))*np.sin(2*np.pi*r2)


def pGauss(X):
    D = len(X)
    return np.exp(-np.dot(X, X) / 2 - np.log(2 * np.pi) * D / 2)
    # return (1/np.sqrt(2*np.pi)**D)*np.exp(-(1/2)*np.dot(X.T, X))


def pltshow(X, Y, Z):
    ax = Axes3D(plt.figure())
    ax.plot_surface (X,Y,Z,cmap='jet',shade=True, linewidth=0.3,edgecolor='black')
    plt.show()

def main():
    [xmin, xmax] = [-4, 4]
    N = 25

    Z = []
    x = np.linspace(xmin, xmax, N)
    X, Y = np.meshgrid(x, x)

    for xx in x:
        for yy in x:
            Z.append(pGauss([xx, yy]))
    Z = np.array(Z).reshape(N, N)

    pltshow(X, Y, Z)


if __name__ == "__main__":
    main()
