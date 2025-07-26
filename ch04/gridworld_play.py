import sys
sys.path.append('..')
import numpy as np
from common.gridworld import GridWorld


env = GridWorld()
V = {}

for state in env.states():
    V[state] = np.random.randn()
env.render_v(V)
