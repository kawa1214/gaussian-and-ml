'''
D = {(3,2), (2, 4), (-1, 1)}
$$
{
    \begin{pmatrix}
    N & \sum_{n=1}^{N}{x_n}\\
    \sum_{n=1}^{N}{x_n} & \sum_{n=1}^{N}{x^2_n}
    \end{pmatrix}
    \begin{pmatrix}
    a\\
    b
    \end{pmatrix}
    =
    \begin{pmatrix}
    \sum_{n=1}^{N}{y_n}\\
    \sum_{n=1}^{N}{x_ny_n}
    \end{pmatrix}
}
$$
'''

import numpy as np
import matplotlib.pyplot as plt

D = ([3, 2], [2, 4], [-1, 1])

N = len(D)
Xn = 0
XXn = 0
Yn = 0
XnYn = 0

for n in D:
    Xn += n[0]
    XXn += n[0]*n[0]
    Yn += n[1]
    XnYn += n[0]*n[1]


arr1 = np.array([[N, Xn], [Xn, XXn]])
arr2 = np.array([[Yn], [XnYn]])

(a, b) = np.dot(np.linalg.inv(arr1), arr2)

print(a, b)

# plot
plt.xlim(-2, 4)
plt.ylim(-1, 5)

plt.plot([-2,4],[0,0],'-',color='#000000',lw=1)
plt.plot([0,0],[-1,5],'-',color='#000000',lw=1)

for n in D:
    plt.plot(n[0],n[1],marker='x', color='#40AAEF')

x = np.arange(-2, 5)

y = a + b*x

# 直線をプロット
plt.plot(x, y, color='#FBA848')

plt.show()