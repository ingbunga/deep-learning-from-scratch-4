import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, arms=10):    # 팔은 슬롯머신의 래버의 은유임. 즉 슬롯머신 대수
        self.rates = np.random.rand(arms)   # 승률 설정
    
    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]
    
    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.Qs))  # 탐색 
        return np.argmax(self.Qs)   # 활용


if __name__ == "__main__":
    runs = 200
    steps = 1000
    epsilon = 0.1
    all_rates = np.zeros((runs, steps))

    for run in range(runs):
        bandit = Bandit()
        agent = Agent(epsilon)
        total_reward = 0
        rates = []          # 승률

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))
        
        all_rates[run] = rates
    
    avg_rates = np.average(all_rates, axis=0)

    # 그래프 그리기: 단계별 보상 총합
    plt.ylabel('Total reward')
    plt.xlabel('Steps')
    plt.plot(avg_rates)
    plt.show()
