import sys
sys.path.append("..")
from collections import defaultdict, deque
import numpy as np

from common.gridworld import GridWorld
from common.utils import greedy_probs

class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_action = {0: 0.25, 1:0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_action)
        self.b = defaultdict(lambda: random_action)
        self.Q = defaultdict(lambda: 0.0)

    def get_action(self, state):
        action_prob = self.b[state]
        actions = list(action_prob.keys())
        probs = list(action_prob.values())
        return np.random.choice(actions, p=probs)

    def update(self, state, action, reward, done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)
        
        # Q 함수 갱신
        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        # 행동 정책과 대상 정책 갱신
        self.pi[state] = greedy_probs(self.Q, state, epsilon=0)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)


if __name__ == '__main__':
    env = GridWorld()
    agent = QLearningAgent()

    EPISODE_NUM = 1_0000

    for episode in range(EPISODE_NUM):
        state = env.reset()
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, done)
            state = next_state

    env.render_q(agent.Q)