import numpy as np
import matplotlib.pyplot as plt

EPSILON = 1e-6

def f(t, w):
    y = w[0]
    u = w[1]
    return np.array([u, u**2 / (y-2)])



def runge_kutta_step(f, h, t, w):

    k1 = f(t, w)
    k2 = f(t + h/2, w + h * k1/2)
    k3 = f(t + h/2, w + h * k2/2)
    k4 = f(t + h, w + h * k3)
    w_new = w + h * (k1 + 2* k2 + 2*k3 + k4) / 6

    return w_new


def get_solution(f, h, t0, tn, w0):
    n = int((tn - t0) / h)
     # Initialize arrays for t and w
    t = np.zeros(n+1)
    w = np.zeros((n+1, 2))
    w[0] = w0


    ab = np.zeros((n+1, 2))
    ab[0] = np.array([0, 1])

    for i in range(n):
        w[i+1] = runge_kutta_step(f, h, t[i], w[i])


        yi, ui = w[i]
        df_dy = (-ui**2)/(yi - 2)**2
        df_du = 2*ui / (yi - 2)

        def g(t, ab):
            A = ab[0]
            B = ab[1]
            return np.array([B, A*df_dy + B*df_du])

        ab[i+1] = runge_kutta_step(g, h, t[i], ab[i])
        
        t[i+1] = t[i] + h

    A, B = ab.T
    return (w, A[-1])


def shooting(f, h, t0, tn, y0, alpha0):
    w0 = np.array([y0, alpha0])

    res = get_solution(f, h, t0, tn, w0)
    w = res[0]
    y, u = w.T

    df = res[1]
    y_L = y[-1]

    prev = 0
    next = alpha0
    
    while (abs(y_L) > EPSILON):
        prev = next
        next = prev - y_L/df
        
        res = get_solution(f, h, t0, tn, np.array([y0, next]))
        y, u = res[0].T

        df = res[1]
        y_L = y[-1]
    
    return y, next


x0 = 0
xn = 1
h = 0.01
y0 = 1.5

alpha0 = -1
y, alpha = shooting(f, h, x0, xn, y0, alpha0)

x = np.linspace(x0, xn, int((xn-x0)/h+1))
plt.plot(x, y)
plt.title('Решение')
plt.ylabel('y')
plt.xlabel('x')
plt.grid()
plt.show()
print('alpha = ' + str(alpha))