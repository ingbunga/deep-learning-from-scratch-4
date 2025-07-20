import numpy as np
import matplotlib.pyplot as plt


class NonStatBandit:
    def __init__(self, arms=10) -> None:
        self.arms = arms
        self.rates = np.random.rand(arms)
    
    def play(self, arm):
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.rand(self.arms)   # 노이즈 추가
        if rate > np.random.rand():
            return 1
        else:
            return 0


class AlphaAgent:
    def __init__(self, epsilon, alpha, actions=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(actions)
        self.alpha = alpha

    def update(self, action, reward):
        # alpha로 갱신
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha
    
    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)
    

if __name__ == "__main__":
    runs = 200
    steps = 1000
    epsilon = 0.1
    all_rates = np.zeros((runs, steps))

    for run in range(runs):
        bandit = NonStatBandit()
        agent = AlphaAgent(epsilon, 0.8)
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