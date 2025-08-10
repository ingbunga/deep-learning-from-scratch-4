import numpy as np
from dezero import Variable
import dezero.layers as L
import dezero.functions as F
from dezero import Model, optimizers
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

lr = 0.2
iters = 10000


class TwoLayerNet(Model):
    def __init__(self, hidden_size: int, out_size: int):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, inputs):
        y = self.l1(inputs)
        y = F.sigmoid(y)
        y = self.l2(y)
        return y


if __name__ == '__main__':
    model = TwoLayerNet(hidden_size=10, out_size=1)
    optimizer = optimizers.SGD(lr)
    optimizer.setup(model)

    for i in range(iters):
        y_pred = model(x)
        loss = F.mean_squared_error(y_pred, y)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        if i % 1000 == 0:
            print(f"Iteration {i}, Loss: {loss.data}")

    print(model.l1.W.data, model.l1.b.data, model.l2.W.data, model.l2.b.data)

    # draw plot
    plt.scatter(x.data, y.data, label='true')
    x_np = x.data.ravel()
    y_pred_np = y_pred.data.ravel()
    sort_index = np.argsort(x_np)
    plt.plot(x_np[sort_index], y_pred_np[sort_index], label='pred', color='red')
    plt.legend()
    plt.show()
