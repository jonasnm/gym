"""
OPENAI gym environment for the Hovorka model using the cambridge parameter set
The reason we are doing this is that the cambridge model is very slow!
This is the base class for the Hovorka models.
    - Actions runs for a longer interval (default 30 mins)
    to get closer to a markov decision process.
    - The model includes meals
    - Default 34 dim observation space (30 min BG and last four actions)
    - Default action space 0 to 50 mU/min of insulin
    - Rendering disabled by default
    - Initialization and reset: Random initialization and no meals!
    - TODO:
        - Double check parameters - renal threshold et al
        - Check difference in previous insulin compared to cambridge model
        - check initial basal rates
"""
# import numpy as np
# np.random.seed(1)


import logging
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.diabetes.meal_generator.meal_generator import meal_generator

import numpy as np

# Plotting for the rendering
import matplotlib.pyplot as plt

# Hovorka simulator
from gym.envs.diabetes.hovorka_model import hovorka_parameters, hovorka_model, hovorka_model_tuple
# from gym.envs.diabetes.reward_function import calculate_reward
from gym.envs.diabetes.hovorka_cambride_pars import hovorka_cambridge_pars
from gym.envs.diabetes.reward_function import RewardFunction

# ODE solver stuff
from scipy.integrate import ode
from scipy.optimize import fsolve

logger = logging.getLogger(__name__)

rewardFunction = RewardFunction()

class HovorkaCambridgeBase(gym.Env):
    # TODO: fix metadata??
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        """
        Initializing the simulation environment.
        """
        # np.random.seed(1) ### Fixing seed

        self.previous_action = 0


        # Bolus carb factor -- [g/U]
        self.bolus = 30

        ## Updating variable parameters -- reward and initial basal rate
        reward_flag, bg_init_flag = self._update_parameters()


        # Action space
        self.sensor_noise = np.random.randn(1)

        # Model parameters
        P = hovorka_parameters(70)
        init_basal_optimal = 6.43
        self.P = P

        # This is the optimal basal rate needed to keep the patient at a steady state
        self.init_basal_optimal = init_basal_optimal

        # This is the space of allowable actions -- from 0 insulin (stop the pump) to twice the basal rate
        self.action_space = spaces.Discrete(3)
        #self.action_space = spaces.Box(0, 2*self.init_basal_optimal, (1,), dtype=np.float32)

        # Initialize episode randomly or at a fixed BG level
        if bg_init_flag == 'random':
            self.init_basal = np.random.choice(np.linspace(init_basal_optimal-2, init_basal_optimal, 10))
        elif bg_init_flag == 'fixed':
            self.init_basal = init_basal_optimal

        # Flag for manually resetting the init
        self.reset_basal_manually = None

        self._seed()
        self.viewer = None

        # ==========================================
        # Setting up the Hovorka simulator
        # ==========================================

        # Initial values for parameters
        initial_pars = (self.init_basal, 0, P)

        # Initial value
        X0 = fsolve(hovorka_model_tuple, np.zeros(11), args=initial_pars)
        self.X0 = X0

        # Simulation setup
        self.integrator = ode(hovorka_model)
        self.integrator.set_integrator('vode', method='bdf', order=5)

        self.integrator.set_initial_value(X0, 0)

        # Simulation time in minutes
        self.simulation_time = 30
        self.n_solver_steps = 1
        self.stepsize = int(self.simulation_time/self.n_solver_steps)

        # Observation space -- the state space for the RL algorithm -> 30 mins of glucose values and 4 insulin values (last 2 hours)
        self.observation_space = spaces.Box(0, 500, (int(self.stepsize + 4),), dtype=np.float32)

        # State is BG, simulation_state is parameters of hovorka model

        # The initial value of insulin is just 4 copies of the basal rate
        initial_insulin = np.ones(4) * self.init_basal_optimal
        initial_bg = X0[-1] * 18
        self.state = np.concatenate([np.repeat(initial_bg, self.simulation_time), initial_insulin])

        self.simulation_state = X0

        # Keeping track of entire blood glucose level for each episode
        self.bg_history = []
        self.insulin_history = initial_insulin

        # ====================
        # Meal setup
        # ====================

        # Meals are loaded from a different file -- the default is four meals with random timing and random carb counting errors

        eating_time = 1
        meals, meal_indicator = meal_generator(eating_time=eating_time, premeal_bolus_time=0)

        # TODO: Clean up these
        self.meals = meals
        self.meal_indicator = meal_indicator
        self.eating_time = eating_time

        # Counter for number of iterations
        self.num_iters = 0

        # If blood glucose is less than zero, the simulator is out of bounds.
        self.bg_threshold_low = 0
        self.bg_threshold_high = 500

        # TODO: This number is arbitrary -- the max length of the episode
        # self.max_iter = 1440
        self.max_iter = 2160

        # Reward flag
        self.reward_flag = reward_flag

        self.steps_beyond_done = None


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _update_parameters(self):
        ''' Update parameters of model,
        this is only used for inherited classes'''

        reward_flag = 'asymmetric'
        #reward_flag = 'gaussian'
        
        bg_init_flag = 'random'

        return reward_flag, bg_init_flag


    def step(self, action):
        """
        Take action. In the diabetes simulation this means increase, decrease or do nothing
        to the insulin to carb ratio (bolus).
        """

        # Manually checking and forcing the action to be within bounds insted of using assert.
        # We should be careful with this
        
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        if action == 0:
              action = 0
        elif action == 1:
              action = self.init_basal_optimal
        elif action == 2:
              action = 5*self.init_basal_optimal
        
# =============================================================================
#         if action > self.action_space.high:
#             action = self.action_space.high
#         elif action < self.action_space.low:
#             action = self.action_space.low
# =============================================================================

        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        self.integrator.set_initial_value(self.simulation_state, self.num_iters)

        bg = []

        # ==========================
        # Integration loop
        # ==========================

        for i in range(self.simulation_time):

            # ===============================================
            # Solving one step of the Hovorka model
            # ===============================================

            # Insulin is basal rate(action) plus a bolus if there is a meal
            insulin_rate = action + (self.meal_indicator[self.num_iters] * (180/self.bolus))

            # Setting the carb and insulin parameter in the model
            self.integrator.set_f_params(insulin_rate, self.meals[self.num_iters], self.P)

            # solving the equations for 1 minute at a time
            self.integrator.integrate(self.integrator.t + 1)

            bg.append(self.integrator.y[-1] * 18)

            self.num_iters += 1

        # Updating environment parameters
        self.simulation_state = self.integrator.y

        # Recording bg history for plotting
        self.bg_history = np.concatenate([self.bg_history, bg])
        self.insulin_history = np.concatenate([self.insulin_history, np.array([insulin_rate])])

        # Updating state (bg and insulin)
        self.state = np.concatenate([bg, list(reversed(self.insulin_history[-4:]))])

        #Set environment done = True if blood_glucose_level is negative or max iters is overflowed
        done = 0

        if (np.max(bg) > self.bg_threshold_high or np.max(bg) < self.bg_threshold_low):
            done = 1

        if self.num_iters > self.max_iter:
            done = 1

        done = bool(done)

        # ====================================================================================
        # Calculate Reward  (and give error if action is taken after terminal state)
        # ====================================================================================

        if not done:
            if self.reward_flag != 'gaussian_with_insulin':
                reward = rewardFunction.calculate_reward(np.array(bg), self.reward_flag, 108)
            else:
                reward = rewardFunction.calculate_reward(np.array(bg), 'gaussian_with_insulin', 108, action)

        elif self.steps_beyond_done is None:
            # Blood glucose below zero -- simulation out of bounds
            self.steps_beyond_done = 0
            # reward = 0.0
            # reward = -1000
            if self.reward_flag != 'gaussian_with_insulin':
                reward = rewardFunction.calculate_reward(np.array(bg), self.reward_flag, 108)
            else:
                reward = rewardFunction.calculate_reward(np.array(bg), 'gaussian_with_insulin', 108, action)
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = -1000

        self.previous_action = action

        return np.array(self.state), np.mean(reward), done, {}


    def reset(self):
        '''
        This is basically a copy of the init function
        '''


        # re init -- in case the init basal has been changed
        if self.reset_basal_manually is None:
            self.init_basal = np.random.choice(np.linspace(self.init_basal_optimal-2, self.init_basal_optimal, 10))
        else:
            self.init_basal = self.reset_basal_manually

        P = self.P
        initial_pars = (self.init_basal, 0, P)

        X0 = fsolve(hovorka_model_tuple, np.zeros(11), args=initial_pars)
        self.X0 = X0
        self.integrator.set_initial_value(self.X0, 0)

        # State is BG, simulation_state is parameters of hovorka model
        initial_bg = X0[-1] * 18
        initial_insulin = np.ones(4) * self.init_basal_optimal
        self.state = np.concatenate([np.repeat(initial_bg, self.stepsize), initial_insulin])

        self.simulation_state = X0
        self.bg_history = []
        self.insulin_history = initial_insulin

        self.num_iters = 0

        eating_time = 1
        meals, meal_indicator = meal_generator(eating_time=eating_time, premeal_bolus_time=0)


        # changing observation space if simulation time is changed
        # if self.simulation_time != 30:
        # if self.stepsize != 1:
            # observation_space_shape = int(self.stepsize + 4 + 1)
            # self.observation_space = spaces.Box(0, 500, (observation_space_shape,), dtype=np.float32)


        self.steps_beyond_done = None
        return np.array(self.state)


    def render(self, mode='human', close=False):
        #TODO: Clean up plotting routine -- this is messy!

        # return None
        if mode == 'rgb_array':
            return None
        elif mode is 'human':
            if not bool(plt.get_fignums()):
                plt.ion()
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111)
                # self.line1, = ax.plot(self.bg_history)
                self.ax.plot(self.bg_history)
                plt.show()
            else:
                # self.line1.set_ydata(self.bg_history)
                # self.fig.canvas.draw()
                self.ax.clear()
                self.ax.plot(self.bg_history)

            plt.pause(0.0000001)
            plt.show()

            return None
        else:
            super(HovorkaCambridgeBase, self).render(mode=mode) # just raise an exception

            plt.ion()
            plt.plot(self.bg_history)
