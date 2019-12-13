import numpy as np
import matplotlib.pyplot as plt


def RBFKernel(X1, X2, Theta1, Theta2):
    return Theta1*np.exp(-((abs(X1-X2)**2)/Theta2))


def plot(X, Y):
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

    plt.plot([-4, 4], [0, 0], '-', color='#000000', lw=1)
    plt.plot([0, 0], [-4, 4], '-', color='#000000', lw=1)

    plt.plot(X, Y, color='r', label='spline')

    plt.show()


def main():
    N = 100
    Theta1 = 1
    Theta2 = 1
    X = np.linspace(-4, 4, N)

    X1, X2 = np.meshgrid(X, X)
    Kernel = RBFKernel(X1, X2, Theta1, Theta2)
    Y = np.random.multivariate_normal(np.zeros(N), Kernel)
    print(np.zeros(N), Kernel.shape)

    plot(X, Y)


if __name__ == "__main__":
    main()
