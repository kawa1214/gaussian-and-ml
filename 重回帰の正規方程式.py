'''
${D = ((1,2),4),((-1,1),2)((3,0),1),((-2,-2),-1)}$
$${
    \text{w} = (\text{X}^T\text{X})^{-1}\text{X}^T\text{y}
}$$
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


D = [[[1, 2], 4], [[-1, 1], 2], [[3, 0], 1], [[-2, -2], -1]]

X = []
y = []

for n in D:
    X.append([1, n[0][0], n[0][1]])
    y.append(n[1])

X = np.array(X)
y = np.array(y)

w = np.dot(np.dot(X.T, y.T), np.linalg.inv(np.dot(X.T, X)))

print(w)

# plot
ax = Axes3D(plt.figure())
ax.set_xlim3d(-3, 4)
ax.set_ylim3d(-3, 4)
ax.set_zlim3d(-3, 4)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")

w0 = []
w1 = []
w2 = []

for n in D:
    ax.plot([n[0][0]], [n[0][1]], [n[1]], marker='x', color='#40AAEF')
    w0.append(n[0][0])
    w1.append(n[0][1])
    #w2.append(w[0]-w[1]*n[0][0]-w[2]*n[0][1])
    print(n[0][0], n[0][1], w[0]-w[1]*n[0][0]-w[2]*n[0][1])

w0 = np.array(w0)
w1 = np.array(w1)

W0, W1 = np.meshgrid(w0, w1)

W2 = 1.202 - W0 + W1

ax.plot_surface(W0, W1, W2, linewidth=0.3, color='#FBA848')

plt.show()
