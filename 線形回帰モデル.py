import numpy as np
import matplotlib.pyplot as plt


data = [[2.7613622, 0.7812694], [-2.5020371, 0.5784024], [-0.6534198, -0.8364839], [-0.5093708, -1.0659939], [1.0698123, -0.5053178], [1.2444854, 0.0656932], [-0.1583863, -1.7132896], [-1.8188962, 0.153927], [-3.6129367, 1.0440645], [-2.8263055, 0.7741641], [-1.0204458, -0.8304516], [0.2611304, -1.3202885], [-0.9667795, -1.0839772], [0.4701717, -0.0456008], [-0.7279929, -0.036916], [0.3050133, 
-1.1703207], [0.3901433, -0.7787978], [3.0430492, 1.2799983], [-1.6307559, -0.2210154], [-3.5160842, 0.572424], [-6.4898371, -0.4126926], [1.8201852, 0.9120053], [-0.7767629, -0.8371592], [-3.4106994, 0.6317732], [-0.9195992, -1.2476601], [-1.1248267, -0.4371627], [-1.5116412, -0.7774516], [1.8203302, 1.2516907], [-1.1185539, -1.6227401], [0.0985207, -0.0693066]]

def linearmodel(Data):
    x = []
    phi = []
    y = []
    for D in Data:
        x.append(D[0])
        y.append(D[1])
    
    for X in x:
        phi.append([1, X, X**2, np.sin(X), np.cos(X)])
    
    x = np.array(x)
    phi = np.array(phi)
    y = np.array(y)

    w = np.dot((np.linalg.inv(np.dot(phi.T, phi))), np.dot(phi.T, y))
    print(w)
    return w

def plot(w, Data):

    plt.xlim(-4, 4)
    plt.ylim(-2, 2)

    plt.plot([-4,4],[0,0],'-',color='#000000',lw=1)
    plt.plot([0,0],[-2,2],'-',color='#000000',lw=1)

    for D in Data:
        plt.plot(D[0], D[1], marker='x', color='#40AAEF')

    x = np.linspace(-4, 6, 100)
    y = w[0] + w[1]*x + w[2]*x**2 + w[3]*np.sin(x) + w[4]*np.cos(x)

    plt.plot(x, y, "r-", color='#FBA848')

    plt.show()

def main ():
    w = linearmodel(data)
    plot(w, data)

if __name__ == "__main__":
    main ()
