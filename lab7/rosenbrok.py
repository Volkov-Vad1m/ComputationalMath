import numpy as np
import matplotlib.pyplot as plt

mu = 10
u0 = np.array([0, 0.001])
t0 = 0
tk = 100
# u = [x, y]
def f(t, u):
    x = u[0]
    y = u[1]
    return np.array([y, mu*(1-y*y)*y - x])

p1 = 0.435866521508459
p2 = 0.4782408332745185
p3 = 0.0858926452170225

a = p1

b21 = p1
b31 = p1
b32 = -2.116053335949811

# f = [f1, f2] ; u = [x, y]
def J(u):
    x = u[0]
    y = u[1]
    
    f1_x = 0
    f1_y = 1
    f2_x = -1
    f2_y = mu*(1 - 3*y*y)

    return np.matrix([[f1_x, f1_y], [f2_x, f2_y]])


def D(u, h):
    return np.eye(2) + a * h * J(u)


def rosenbrok(f, h, t0, tn, u0):

    # Define the number of steps
    n = int((tn - t0) / h)

    # Initialize arrays for t and w
    t = np.zeros(n+1)
    u = np.zeros((n+1, 2))

    t[0] = t0
    u[0] = u0

    for i in range(n):
        k1 = np.linalg.solve(D(u[i],h), h * f(t[i], u[i]))
        k2 = np.linalg.solve(D(u[i],h), h * f(t[i], u[i] + b21 * k1))
        k3 = np.linalg.solve(D(u[i],h), h * f(t[i], u[i] + b31 * k1 + b32 * k2))
        u[i+1] = u[i] + p1*k1 + p2*k2 + p3*k3
        t[i+1] = t[i] + h

    return t, u    



h = 0.001
t, u = rosenbrok(f, h, t0, tk, u0)
x, y = map(np.array, zip(*u))

plt.plot(x, y)
plt.xlabel('t')
plt.ylabel('x')
plt.show()


plt.plot(t, x)
plt.title('x(t)')
plt.xlabel('t')
plt.ylabel('x')
plt.show()

