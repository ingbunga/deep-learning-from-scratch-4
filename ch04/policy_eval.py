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


if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0.0)

    V = policy_eval(pi, V, env, gamma)
    env.render_v(V, pi)
