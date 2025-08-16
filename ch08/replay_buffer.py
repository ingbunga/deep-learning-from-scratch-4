from collections import deque
import random
import numpy as np
import gym

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)
    
    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([d[0] for d in data])
        action = np.array([d[1] for d in data])
        reward = np.array([d[2] for d in data])
        next_state = np.stack([d[3] for d in data])
        done = np.array([d[4] for d in data])

        return state, action, reward, next_state, done
    

if __name__ == "__main__":
    env = gym.make('CartPole-v0', render_mode='human')
    replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=32)

    for episode in range(10):
        state = env.reset()[0]
        done = False

        while not done:
            action = 0
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
        
    state, action, reward, next_state, done = replay_buffer.get_batch()
    print("Sampled batch from replay buffer:")
    print(state.shape)
    print(action.shape)
    print(reward.shape)
    print(next_state.shape)
    print(done.shape)