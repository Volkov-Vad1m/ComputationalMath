import numpy as np
import matplotlib.pyplot as plt
import math


x0 = 0
xL = 1

#Пусть сетка будет из 100000
L = 100000
x = np.linspace(x0, xL, L)

x_raz = 0.525 



def k(x):
    if(x < x_raz):
        return x + 1
    else:
        return x

def q(x):
    return math.exp(-x)
    
def f(x):
    if(x < x_raz):
        return 1
    else:
        return x**3
    

h = (xL - x0) / (L-1)
l_alpha = int(x_raz / h)
l_beta = l_alpha + 1

def get_coeffs_reduced_equation(l_alpha, l_beta, x0, xL, L):
    h = (xL - x0) / (L-1)

    a = np.zeros(L)
    b = np.zeros(L)
    c = np.zeros(L)
    d = np.zeros(L)

    for l in range(1, l_alpha):
        a[l] = k(l*h + h/2)
        b[l] = -( k(l*h + h/2) + k(l*h - h/2) + q(l*h) * h * h )
        c[l] = k(l*h - h/2)
        d[l] = -f(l* h) * h * h

    for l in range(l_beta + 1, L-1):
        a[l] = k(l*h + h/2)
        b[l] = -( k(l*h + h/2) + k(l*h - h/2) + q(l*h) * h * h )
        c[l] = k(l*h - h/2)
        d[l] = -f(l*h) * h * h
    
    return a, b, c, d

def get_run_through_coeffs(a, b, c, d, L, u0, uL):

    alpha = np.zeros(L)
    beta  = np.zeros(L)

    alpha[1] = -a[1] / b[1]
    beta[1]  = (d[1] - c[1] * u0) / b[1]

    alpha[L - 2] = -c[L - 2] / b[L - 2]
    beta[L - 2]  = (d[L - 2] - c[L - 2] * uL) / b[L - 2]

    for l in range(2, l_alpha):
        alpha[l] = -a[l] / (b[l] +c[l] * alpha[l - 1])
        beta[l]  = (d[l] - c[l] * beta[l - 1]) / (b[l] + c[l] *alpha[l - 1])

    for l in range(L - 3, l_beta, -1):
        alpha[l] = -c[l] / (b[l] + a[l] * alpha[l + 1])
        beta[l]  = (d[l] - a[l] * beta[l + 1]) / (b[l] + a[l] * alpha[l + 1])
    
    return alpha, beta

def get_solution(l_alpha, l_beta, x0 = 0, xL = 1, L = 100000, u0 = 0, uL = 1):

    a, b, c, d = get_coeffs_reduced_equation(l_alpha, l_beta, x0, xL, L)
    alpha, beta = get_run_through_coeffs(a, b, c, d, L, u0, uL)

    u = np.zeros(L)

    u[0] = u0
    u[L-1] = uL

    u[l_alpha] = (k(l_alpha*h) * beta[l_alpha - 1] + k(l_beta*h) *beta[l_beta + 1]) / ( k(l_alpha*h) * (1 - alpha[l_alpha - 1]) + k(l_beta*h) * (1 - alpha[l_beta + 1]) )
    u[l_beta] = u[l_alpha]

    u[l_alpha - 1] = alpha[l_alpha - 1] * u[l_alpha] + beta[l_alpha - 1]
    u[l_beta + 1] = alpha[l_beta + 1] * u[l_beta] + beta[l_beta + 1]

    for l in range(l_alpha - 1, 0, -1):
        u[l] = alpha[l] * u[l + 1] + beta[l]

    for l in range(l_beta + 1, L-1):
        u[l] = alpha[l] * u[l - 1] + beta[l]

    return u




u = get_solution(l_alpha, l_beta)

plt.plot(x, u)
plt.title("u(x)")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid()
plt.show()