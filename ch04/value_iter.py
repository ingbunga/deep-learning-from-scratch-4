import sys
sys.path.append('..')
from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld
from ch04.policy_iter import greedy_policy

def argmax(d: dict[int, float]) -> int:
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key
            break
    return max_key

def value_iter_onestep(V, env: GridWorld, gamma: float):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_values = {}
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            action_values[action] = r + gamma * V[next_state]

        V[state] = max(action_values.values())
    return V

def value_iter(V, env: GridWorld, gamma: float, threshold=1e-3, is_render=False):
    while True:
        if is_render:
            env.render_v(V)
        
        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)

        delta = max(abs(V[state] - old_V[state]) for state in env.states())
        if delta < threshold:
            break
    return V


if __name__ == "__main__":
    V = defaultdict(lambda: 0.0)
    env = GridWorld()
    gamma = 0.9

    V = value_iter(V, env, gamma, is_render=True)

    pi = greedy_policy(V, env, gamma)
    env.render_v(V, pi)
