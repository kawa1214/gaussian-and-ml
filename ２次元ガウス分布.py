import numpy as np
import matplotlib.pyplot as plt
import random


def Gauss():
    r1 = random.random()
    r2 = random.random()
    return np.sqrt(-2*np.log(r1))*np.sin(2*np.pi*r2)


def pGauss(Sigma):
    g = Gauss()
    return np.dot(Sigma, g)


def pltshow(XY):

    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

    plt.plot([-4,4],[0,0],'-',color='#000000',lw=1)
    plt.plot([0,0],[-4,4],'-',color='#000000',lw=1)

    for xy in XY:
        print(xy[0][0], xy[1][0])
        plt.plot(xy[0][0], xy[1][0], marker='o', color='#40AAEF')

    plt.show()

def main():
    N = 100
    XY = []
    Sigma = [[1, 0.9], [0.9, 1]]
    Sigma = np.array(Sigma)

    for i in range(N):
        XY.append(pGauss(Sigma))
        #print(pGauss(Sigma))

    #print(XY)
    pltshow(XY)


if __name__ == "__main__":
    main()
