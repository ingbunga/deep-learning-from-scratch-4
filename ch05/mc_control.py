import sys
sys.path.append('..')
from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld


def greedy_probs(Q, state, epsilon=0.0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action: int = np.argmax(qs) # type: ignore

    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += (1.0 - epsilon)
    return action_probs


class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.alpha = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0.0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)
    
    def reset(self):
        self.memory.clear()

    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = reward + self.gamma * G
            key = (state, action)
            self.Q[key] += (G - self.Q[key]) * self.alpha
            self.pi[state] = greedy_probs(self.Q, state, self.epsilon, self.action_size)


if __name__ == '__main__':
    env = GridWorld()
    agent = McAgent()

    for episode in range(10000):
        state = env.reset()
        agent.reset()
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.add(state, action, reward)
            state = next_state

        agent.update()
    
    # 가치 함수 시각화
    env.render_q(agent.Q)