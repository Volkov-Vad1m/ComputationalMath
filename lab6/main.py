import numpy as np
import matplotlib.pyplot as plt

# w - array
def f(t, w):
    x = w[0]
    y = w[1]
    z = w[2]
    u = w[3]
    return np.array([z, u, -x/pow(x**2 + y**2, 3/2), -y/pow(x**2 + y**2, 3/2)])


def runge_kutta(f, h, t0, tn, w0):

     # Define the number of steps
    n = int((tn - t0) / h)
     # Initialize arrays for t and w
    t = np.zeros(n+1)
    w = np.zeros((n+1, 4))


    t[0] = t0
    w[0] = w0

    for i in range(n):
        k1 = f(t[i], w[i])
        k2 = f(t[i] + h/2, w[i] + h * k1/2)
        k3 = f(t[i] + h/2, w[i] + h * k2/2)
        k4 = f(t[i] + h, w[i] + h * k3)
        w[i+1] = w[i] + h * (k1 + 2* k2 + 2*k3 + k4) / 6
        t[i+1] = t[i] + h

    return t, w


h = 0.01
w0 = np.array([0.5 , 0, 0, 1.73])
t0 = 0
tn = 20

t, w  = runge_kutta(f, h, t0, tn, w0)

print(w)
# x, y, z, u = [arr[0] for arr in w], [arr[1] for arr in w], [arr[2] for arr in w], [arr[3] for arr in w]
x, y, z, u = map(np.array, zip(*w))

plt.plot(x, y)
plt.title('фазовая траектория')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

def addams(f, h, t0, tn, w0):
    n = int((tn - t0) / h)

    t = np.zeros(n+1)
    w = np.zeros((n+1, 4))

    if n < 3:
        raise Exception("must be at least 4 points")
    
    _ ,addams_init = runge_kutta(f, h, t0, t0 +3*h, w0)
 
    for i in range(len(addams_init)):
        w[i] = addams_init[i]
        print(addams_init[i])
    


    for i in range(n-3):
        w[i+4] = w[i+3] + h * (55/24*f(t[i+3], w[i+3]) - 59/24*f(t[i+2], w[i+2]) + 37/24*f(t[i+1], w[i+1]) - 3/8*f(t[i], w[i]))
        t[i+4] = t[i+3] + h

    return t, w

h = 0.01
w0 = np.array([0.5 , 0, 0, 1.73])
t0 = 0
tn = 20

t, w  = addams(f, h, t0, tn, w0)

# x, y, z, u = [arr[0] for arr in w], [arr[1] for arr in w], [arr[2] for arr in w], [arr[3] for arr in w]
x, y, z, u = map(np.array, zip(*w))

plt.plot(x, y)
plt.title('фазовая траектория')
plt.xlabel('x')
plt.ylabel('y')
plt.show()