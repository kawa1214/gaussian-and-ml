import numpy as np
import matplotlib.pyplot as plt

Data = [[-0.5, 0.7],[0.5,1.8],[1,1.7],[1.4,2.3],[3,1],[2.3,0],[2.5,0.2],[1.5,2],[1.1,2.4],[0.7,1.5]]

# plot parameters
N    = 100
xmin = -1
xmax = 3.5
ymin = -1
ymax = 3

def plotSet(xmin, xmax, ymin, ymax):
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

def plotPoint(X, Y):
    plt.scatter(X, Y, marker='.',color='r')

def plotLine(X, Y):
    plt.plot(X, Y, color='#FBA848', linestyle='solid')

def plotFill(X, Y, Variance):
    plt.fill_between (X, Y - 2*np.sqrt(Variance), Y + 2*np.sqrt(Variance), color='#ccccff')


def GaussKernel(X1, X2, Params):
    [tau, sigma] = Params
    return tau * np.exp (-(X1 - X2)**2 / (2 * sigma * sigma))


def KernelMatrix(X, Kernel, Params):
    X1, X2 = np.meshgrid(X, X)
    N = len(X)
    return Kernel(X1, X2, Params) + eta*np.eye(N)

def AndKernel(xtest, Xtrain, Kernel, Params):
    And = []
    for xtrain in Xtrain:
        And.append(Kernel(xtest, xtrain, Params))
    And = np.array(And)
    return And

def PredictYandVariance(Xtrain, Ytrain, Xtest, Kernel, Params):
    Ytest = []
    Variance = []
    K = KernelMatrix(Xtrain, Kernel, Params)
    Kinv = np.linalg.inv(K)

    for xtest in Xtest:
        s = Kernel(xtest, xtest, Params) + eta
        k = AndKernel(xtest, Xtrain, Kernel, Params)
        Ytest.append(np.dot(np.dot(k.T, Kinv), Ytrain))
        Variance.append(s-np.dot(np.dot(k.T, Kinv),k))
    return Ytest, Variance

    
def GaussGrad(Xtrain, Ytrain, Kernel, Params):
    K = KernelMatrix(Xtrain, Kernel, Params)

    Kinv = np.linalg.inv(K)
    Ytrain = np.array(Ytrain)
    L = - np.log(K) - np.dot(np.dot(Ytrain.T, Kinv), Ytrain)
    print(L)

# GP kernel parameters
eta   = 0.1
tau   = 1
sigma = 1

def main():
    Xtrain = []
    Ytrain = []
    Xtest = np.linspace (xmin, xmax, N)

    for data in Data:
        Xtrain.append(data[0])
        Ytrain.append(data[1])

    Kernel = GaussKernel
    KernelGrad = GaussGrad

    
    Params = [tau, sigma]

    NewParams = KernelGrad(Xtrain, Ytrain, Kernel, Params)




    Ytest, Variance = PredictYandVariance(Xtrain, Ytrain, Xtest, Kernel, Params)

    plotSet(xmin, xmax, ymin, ymax)

    plotFill(Xtest, Ytest, Variance)
    plotPoint(Xtrain, Ytrain)
    plotLine(Xtest, Ytest)
    
    plt.show()


if __name__ == "__main__":
    main()
