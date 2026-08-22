import numpy as np
import matplotlib.pyplot as plt


M = np.array([[1,1,1],[4,2,1],[9,3,1]])

b = np.array([1,2,0])

x = np.linalg.solve(M,b)

a,b,c = x



xx = np.array([1,2,3])
yy = np.array([1,2,0])
x = np.linspace(0,4,100)
f = lambda t: a*t**2+b*t+c
plt.plot(xx,yy, '*')
plt.plot(x,f(x))
plt.show()