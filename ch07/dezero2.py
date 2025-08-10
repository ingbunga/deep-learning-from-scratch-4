import numpy as np
from dezero import Variable
import dezero.functions as F

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

iters = 10000
lr = 0.001

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0**2)**2 + (1 - x0)**2
    return y

for i in range(iters):
    print(x0, x1)
    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()

    y.backward()

    x0.data -= lr * x0.grad.data
    x1.data -= lr * x1.grad.data

print(x0, x1)