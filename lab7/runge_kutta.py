import numpy as np
import matplotlib.pyplot as plt

### певый способ - Рунге Кутты
def f(t, w):
    x = w[0]
    y = w[1]
    return np.array([y, mu*(1-y*y)*y - x])
w0 = np.array([0 , 0.001])
t0 = 0
tn = 1000
mu = 10

c = [1/4, 3/4]
b = [0.5, 0.5]
A = [[1/4, 0], [1/2, 1/2]]


def runge_kutta(f, h, t0, tn, w0):

    n = int((tn - t0) / h)

    t = np.zeros(n+1)
    w = np.zeros((n+1, 2))

    t[0] = t0
    w[0] = w0
    
    
    for i in range(n):
        # method simple iteration 
        k12 = [np.zeros(2), np.zeros(2)]
        
        for j in range(2):
            k12 = [f(t + h * c[k], w[i] + h * sum([x*y for x,y in zip(k12, A[k])])) for k in range(2)]
            
        w[i+1] = w[i] + h * sum([x*y for x,y in zip(k12, b)])
        t[i+1] = t[i] + h

    return t, w


t, w  = runge_kutta(f, 0.001, t0, tn, w0)

x, y = map(np.array, zip(*w))

plt.plot(x, y)
plt.title('фазовая траектория')
plt.xlabel('x')
plt.ylabel('dx/dt')
plt.show()

plt.plot(t, x)
plt.title('x(t)')
plt.xlabel('t')
plt.ylabel('x')
plt.show()