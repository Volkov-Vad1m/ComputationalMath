from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import math
EPSILON = 1E-5

def f(x):
    return x*pow(2,x) - 1

#построим график
x = np.linspace(-1, 1, 1000)
plt.plot(x, f(x))
plt.grid()
plt.show()

#область локализации [0;1]
def phi(x):
    return pow(2, -x)

x = np.linspace (0.01, 1, 1000)
plt.plot(x, abs(phi(x)))
plt.grid()
plt.show()


def msi(x_0):
    iters = 0
    x_cur = x_0
    x_prev = 0

    while(abs(x_cur - x_prev) > EPSILON):
        x_prev = x_cur
        
        x_cur = phi(x_prev)

        iters += 1
        
    return x_cur, iters


x, iters = msi(0.8)
print("Метод простой итерации")
print("x =", x)
print("iters =", iters)
print("Подставим корень в исходное уравнение")
print(f(x))

x = fsolve(f, 0.5)
print("Через fsolve")
print("x =", x[0])



#метод Ньютона 12.5 г
def F_1(x, y):
    return np.sin(x+2) - y - 1.5

def F_2(x, y):
    return np.cos(y-2) + x - 0.5

def revJ_mult_F(x,y):
    det = -np.sin(y-2)*np.cos(x+2) + 1
    row1 = (-np.sin(y - 2) * F_1(x,y) + F_2(x,y) ) / det
    row2 = (-F_1(x,y) + np.cos(x + 2) * F_2(x,y) ) / det
    return np.array([row1, row2])


def newton(x_0, y_0):
    iters = 0
    x_cur = x_0
    y_cur = y_0
    x_prev = 0
    y_prev = 0

    while((abs(x_cur - x_prev) > EPSILON) and (abs(y_cur - y_prev) > EPSILON)):
        x_prev = x_cur
        y_prev = y_cur
        

        arr = revJ_mult_F(x_prev, y_prev)
        x_cur = x_prev - arr[0]
        y_cur = y_prev - arr[1]

        iters += 1

    return x_cur, y_cur, iters

x, y, iters = newton(153232, 15312321)
print("Метод Ньютона")
print("x =", x)
print("y =", y)
print("iters =", iters)

print("Подставим корни в исходную систему")
print(F_1(x,y))
print(F_2(x,y))
