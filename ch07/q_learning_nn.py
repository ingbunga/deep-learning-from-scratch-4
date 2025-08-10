import sys
sys.path.append("..")
import numpy as np
from dezero import Variable
import dezero.layers as L
import dezero.functions as F
from dezero import Model, optimizers
import matplotlib.pyplot as plt
from common.gridworld import GridWorld

np.random.seed(0)
lr = 0.2
iters = 10000

def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)

    y, x = state
    index = y * WIDTH + x
    vec[index] = 1.0

    return vec[np.newaxis, :]


class QNet(Model):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = L.Linear(100)
        self.l2 = L.Linear(4)

    def forward(self, inputs: Variable) -> Variable:
        h = F.relu(self.l1(inputs))
        h = self.l2(h)
        return h    #type: ignore


class QLearningAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = QNet()
        self.optimizer = optimizers.SGD(self.lr)
        self.optimizer.setup(self.qnet)
    
    def get_action(self, state) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)

        qs = self.qnet(state)
        return qs.data.argmax()
    
    def update(self, state, action, reward, next_state, done):
        done = int(done)
        next_qs = self.qnet(next_state)
        next_q = next_qs.max(axis=1)
        next_q.unchain()
        
        target = (1-done) * self.gamma * next_q + reward
        qs = self.qnet(state)
        q = qs[:, action]
        loss = F.mean_squared_error(q, target)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

        return loss.data
    

if __name__ == '__main__':
    env = GridWorld()
    agent = QLearningAgent()

    episodes = 1000
    loss_history = []

    for episode in range(episodes):
        state = env.reset()
        state = one_hot(state)
        total_loss, cnt = 0, 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = one_hot(next_state)

            loss = agent.update(state, action, reward, next_state, done)
            total_loss += loss
            cnt += 1
            state = next_state
        
        average_loss = total_loss / cnt
        loss_history.append(average_loss)
    
    # plot loss_history
    plt.plot(loss_history)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.show()
    # make Q table for visualization
    # dict[((x, y), action)] -> Q-value
    q_table = {}
    for y in range(3):
        for x in range(4):
            state = (y, x)
            q_values = agent.qnet(one_hot(state)).data
            for action in range(4):
                q_table[(state, action)] = q_values[0, action]
    print("Q-table:")
    env.render_q(q_table)
