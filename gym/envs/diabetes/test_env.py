import numpy as np
import gym

env = gym.make('HovorkaCambridge-v0')

reward = []
bg = []
cgm = []

env.env.reset_basal_manually = env.env.init_basal_optimal
env.reset()

for i in range(72):
    # Step for the minimal/hovorka model
    s, r, d, i = env.step(np.array([env.env.init_basal_optimal]))

# env.render()
