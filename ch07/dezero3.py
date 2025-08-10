import numpy as np
from dezero import Variable
import dezero.functions as F

np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

def predict(x):
    y = F.matmul(x, W) + b
    return y


iters = 10000
lr = 0.001

def MSE(x0, x1):
    diff = x1 - x0
    return F.sum(diff**2) / diff.size

for i in range(iters):
    y_pred = predict(x)
    loss = MSE(y_pred, y)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    

print(W, b)