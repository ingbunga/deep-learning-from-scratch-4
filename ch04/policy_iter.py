import sys
sys.path.append('..')
from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld


def eval_onestep(pi, V, env: GridWorld, gamma=0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_probs = pi[state]
        new_V = 0

        for action, action_probs in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            new_V += action_probs * (r + gamma * V[next_state])

        V[state] = new_V
    return V

def policy_eval(pi, V, env: GridWorld, gamma=0.9, threshold=1e-3):
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)

        delta = max(abs(V[state] - old_V[state]) for state in env.states())
        
        if delta < threshold:
            break
    return V

def argmax(d: dict[int, float]) -> int:
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key
            break
    return max_key

def greedy_policy(V, env: GridWorld, gamma=0.9):
    pi = {}

    for state in env.states():
        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            action_values[action] = r + gamma * V[next_state]
        
        max_action = argmax(action_values)
        action_probs = {action: 0.0 for action in env.actions()}
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    return pi

def policy_iter(env: GridWorld, gamma=0.9, threshold=1e-3, is_render=False):
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0.0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi = greedy_policy(V, env, gamma)

        if is_render:
            env.render_v(V, new_pi)
        
        if new_pi == pi:
            break
        pi = new_pi
    return pi


if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9
    pi = policy_iter(env, gamma, is_render=True)
