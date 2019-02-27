import numpy as np
import gym
# import seaborn as sns
# sns.set()

from pylab import plot, figure, title, show, ion, legend
from gym.envs.diabetes.hovorka_cambride_pars import hovorka_cambridge_pars
from gym.envs.diabetes.hovorka_model import hovorka_parameters

# env = gym.make('HovorkaGaussian-v0')
# env = gym.make('HovorkaGaussian-v0')
# np.random.seed(0)
env = gym.make('HovorkaCambridge-v0')

P = hovorka_parameters(70)
env.env.P = P
init_basal_optimal = 6.43
env.env.init_basal_optimal = init_basal_optimal
env.env.reset_basal_manually = init_basal_optimal
# env = gym.make('HovorkaGaussianInsulin-v0')
# env = gym.make('HovorkaBinary-v0')
# env = gym.make('HovorkaAbsolute-v0')
meal_amount = np.array([40, 80, 60, 30])
env.env.meals = np.zeros(1440)
env.env.meals[[8*60, 12*60, 18*60, 22*60]] = np.array([40, 80, 60, 30])
env.env.meals = env.env.meals * 1000/180

env.env.meal_indicator = np.zeros(1440)
env.env.meal_indicator[[8*60, 12*60, 18*60, 22*60]] = np.array([40, 80, 60, 30]) + np.random.randint(-20, 20)
env.env.meal_indicator = env.env.meal_indicator * 1000/180



# basal = 0
# env.env.reset_basal_manually = 6.43

reward = []
bg = []
cgm = []

# env.env.reset_basal_manually = 6.43
env.env.bolus = 40
# env.env.meals = np.zeros(1440)
# env.env.meal_indicator = np.zeros(1440)
env.reset()

for i in range(48):

    # Step for the minimal/hovorka model
    s, r, d, i = env.step(np.array([init_basal_optimal]))
    # s, r, d, i = env.step(np.array([0]))

    bg.append(env.env.simulation_state[4])
    cgm.append(env.env.simulation_state[-1] * env.env.P[12])
    reward.append(r)
    # print(r)

    # Step for the discrete Hovorka incremental
    # env.step(2)



# env.render()
# figure()
# # plot(env.env.bg_history)
# plot(bg)
# plot(cgm)
# legend(['bg', 'cgm'])
# title('bg and cgm')
# ion()
# show()
plot(env.env.bg_history)
show()
