'''
Minimal example to test if the gym works!
Do not edit!
'''
import numpy as np
import gym
import seaborn as sns
sns.set()


env = gym.make('HovorkaCambridge-v0')

init_basal_optimal = 6.43
env.env.init_basal_optimal = init_basal_optimal
env.env.reset_basal_manually = init_basal_optimal

env.reset()

for i in range(48):
    s, r, d, i = env.step(np.array([init_basal_optimal]))


env.render()