import numpy as np
import matplotlib.pyplot as plt

x = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
f = [0.0,  0.021470,  0.293050, 0.494105, 0.541341, 0.516855, 0.468617, 0.416531,0.367879]


plt.scatter(x, f)
plt.show()

def trapezoid_method(x, y, r=1):
    result = 0
    h = x[1] - x[0]
    n = len(x)

    for i in range(0, n-r, r):
        result += ((f[i] + f[i+r]) / 2)
    
    return result * r * h

def simpson_method(x, f):
    n = len(x)
    h = x[1] - x[0]
    result = (h/3) * (f[0] + 2*sum(f[:n-2:2]) + 4*sum(f[1:n-1:2]) + f[n-1])

    return result

p = 2
r = 2

print("Метод трапеции")
I = trapezoid_method(x, f)
print(I)
print("----------------------------------")

print("Метод Симпсона")
Is = simpson_method(x, f)
print(Is)
print("----------------------------------")

print("Метод трапеции c удвоенной сеткой")
Ir = trapezoid_method(x, f, r)
print(Ir)
print("----------------------------------")

print("Метод Рунге")
I_Runge = (pow(r,p) * I - Ir)/(pow(r,p) - 1)
print(I_Runge)
print("----------------------------------")

print("Cравним точности")
print("Трапеция и Симпсон:", abs(Is - I))
print("Трапеция и Рунге: ", abs(I - I_Runge))
print("Рунге и Симпсон: ", abs(I_Runge - Is))