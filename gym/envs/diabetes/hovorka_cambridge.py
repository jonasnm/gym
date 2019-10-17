"""
OPENAI gym environment for the Hovorka model


This is the base class for the Hovorka models.
    - Actions runs for a longer interval (default 30 mins)
    to get closer to a markov decision process.
    - The model includes meals
    - Default 34 dim observation space (30 min BG and last four actions)
    - Default action space 0 to 50 mU/min of insulin
    - Rendering disabled by default

    - Initialization and reset: Random initialization and no meals!


    - TODO:
        - Update this documentation!!!!!
        - Double check parameters - renal threshold et al
        - Check difference in previous insulin compared to cambridge model
        - check initial basal rates

"""

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

        # Fixing the random seed -- for reproducible experiments
        np.random.seed(1)

        # Miguel: Why is this needed?
        self.previous_action = 0

        # Bolus carb factor -- [g/U]
        self.bolus = 30

        ## Loading variable parameters -- used if environment is extended
        reward_flag, bg_init_flag = self._update_parameters()


        # Miguel: CamelCase is not used in rest of code

        # Initialize bolus history -- used for insulin on board
        self.bolusHistoryIndex = 0
        self.bolusHistoryValue = []
        self.bolusHistoryTime = []
        self.insulinOnBoard = np.zeros(1)

        # Initialize sensor model -- Miguel: do we need all of this?
        # self.CGMlambda = 15.96    # Johnson parameter of recalibrated and synchronized sensor error.
        # self.CGMepsilon = -5.471  # Johnson parameter of recalibrated and synchronized sensor error.
        # self.CGMdelta = 1.6898    # Johnson parameter of recalibrated and synchronized sensor error.
        # self.CGMgamma = -0.5444   # Johnson parameter of recalibrated and synchronized sensor error.
        # self.CGMerror = 0
        self.sensor_noise = np.random.randn(1)
        # self.CGMaux = []
        # self.sensorNoiseValue = 0.07 # Set a value


        # Model parameters
        P = hovorka_parameters(70)
        init_basal_optimal = 6.43
        self.P = P
        self.init_basal_optimal = init_basal_optimal

        # Initializing the action space
        self.action_space = spaces.Box(0, 2*self.init_basal_optimal, (1,), dtype=np.float32)

        # Initial basal rate -- used for init and reset. Either randomly initialized or by a fixed value.

        if bg_init_flag == 'random':
            self.init_basal = np.random.choice(np.linspace(init_basal_optimal-2, init_basal_optimal, 10))
        elif bg_init_flag == 'fixed':
            self.init_basal = init_basal_optimal

        # Flag for manually resetting the init when the episode restarts
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

        # Simulation time in minutes -- the default is solving the simulator 30 minutes at a time
        # with a solver time of one minute.
        self.simulation_time = 30
        self.n_solver_steps = 1
        self.stepsize = int(self.simulation_time/self.n_solver_steps)

        # State space for the environment -- [30 min of BG, last 4 insulin action, bolus (if given) during last 30 mins]
        self.observation_space = spaces.Box(0, 500, (int(self.stepsize + 4 + 2),), dtype=np.float32)

        # State is BG, simulation_state is parameters of hovorka model
        initial_bg = X0[-1] * 18
        initial_insulin = np.ones(4) * self.init_basal_optimal
        initial_iob = np.zeros(1)

        # Initial state
        self.state = np.concatenate([np.repeat(initial_bg, self.stepsize), initial_insulin, initial_iob, np.zeros(1)])

        self.simulation_state = X0

        # Keeping track of entire blood glucose level and insulin for each episode
        self.bg_history = []
        self.insulin_history = initial_insulin

        # ====================
        # Meal setup
        # ====================

        # Default eating time is considered one minute -- empirically the simulations are not too sensitive to this choice.
        eating_time = 1

        # Meals are carb intake and meal_indicator is the counted carbs by the patient
        meals, meal_indicator = meal_generator(eating_time=eating_time, premeal_bolus_time=0)

        self.meals = meals
        self.meal_indicator = meal_indicator
        self.eating_time = eating_time

        # Counter for number of iterations
        self.num_iters = 0

        # If blood glucose is less than zero or above 500, the simulator is considered out of bounds.
        self.bg_threshold_low = 0
        self.bg_threshold_high = 500

        # The max episode lenght is 36 hours
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

        # Default reward is the asymmetric reward
        reward_flag = 'asymmetric'

        # bg is randomly initialized per episode by default
        bg_init_flag = 'random'

        return reward_flag, bg_init_flag


    # Miguel: again with CamelCase. Not a big problem, but looks a bit weird.
    def scalableExpIOB(self, t, tp, td):
            #SCALABLEEXPIOB
            # Calculates the insulin bolus on board using a decay
            # expenontiel. Function taken from
            # https://github.com/ps2/LoopIOB/blob/master/ScalableExp.ipynb
            # Original contributor Dragan Maksimovic (@dm61)
            #
            # Inputs:
            #    - t: Time duration after bolus delivery.
            #    - tp: Time of peak action of insulin.
            #    - td: Time duration of insulin action.
            #
            # For more info on tp and td:
            # http://guidelines.diabetes.ca/cdacpg_resources/Ch12_Table1_Types_of_Insulin_updated_Aug_5.pdf
            #

            if t > td:
                iob = 0
                return iob
            else:
                tau = tp * (1 - tp / td) / (1 - 2 * tp / td)
                a = 2 * tau / td
                S = 1 / (1 - a + (1 + a) * np.exp(-td/tau))
                iob = 1 - S * (1 - a) * ((t**2 / (tau * td * (1 - a)) - t / tau - 1) * np.exp(-t/tau) + 1)
                return iob

    def step(self, action):
        """
        Take action. In the diabetes simulation this means increase, decrease or do nothing
        to the insulin to carb ratio (bolus).
        """
        if action > self.action_space.high:
            action = self.action_space.high
        elif action < self.action_space.low:
            action = self.action_space.low

        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        self.integrator.set_initial_value(self.simulation_state, self.num_iters)

        bg = []
        bolus_given = np.zeros(1)

        for i in range(self.simulation_time):

            # ===============================================
            # Solving one step of the Hovorka model
            # ===============================================

            # Calculating insulin on board
            self.insulinOnBoard = np.zeros(1)
            if self.bolusHistoryIndex > 0:
                for b in range(self.bolusHistoryIndex):
                    self.insulinOnBoard = self.insulinOnBoard + self.bolusHistoryValue[b] * self.scalableExpIOB(self.num_iters - self.bolusHistoryTime[b], 75, 300)

            # If there is a meal, give a bolus
            insulin_rate = action + self.bolus

            bolus_given =  np.array([self.bolus])

            # Add given bolus to history
            if self.meal_indicator[self.num_iters] > 0:
                self.bolusHistoryIndex = self.bolusHistoryIndex + 1
                self.bolusHistoryValue.append(self.bolus)
                self.bolusHistoryTime.append(self.num_iters)


            # Updating the carb and insulin parameters in the model
            self.integrator.set_f_params(insulin_rate, self.meals[self.num_iters], self.P)

            # Integration step
            self.integrator.integrate(self.integrator.t + 1)

            # ===============
            # CGM noise -- uncomment if CGM noise is to be addded
            # ===============

            # if i % 5 == 0:
            # # johnson
            #     self.sensor_noise = 0.7 * (self.sensor_noise[0] + np.random.randn(1))
            # # paramMCHO = 180
            #     self.CGMerror = self.CGMepsilon + self.CGMlambda * np.sinh((self.sensor_noise[0] - self.CGMgamma) / self.CGMdelta)
            # # ar(1), colored}
            # if i % 5 == 0:
            # phi = 0.8
            # self.CGMerror = phi * self.CGMerror + np.sqrt(1 - phi ** 2) * self.sensorNoiseValue * np.random.randn(1)[0]

            # # mult
            # self.CGMerror = self.sensorNoiseValue * self.state(self.integrator.y[-1]) * np.random.randn(1)[0]

            # # white, add
            # self.CGMerror = self.sensorNoiseValue * np.random.randn(1)

            # # No noise
            # self.CGMerror = 0

            # self.CGMaux.append(self.CGMerror)

            # bg.append(self.integrator.y[-1] * 18 + self.CGMerror)

            # Stop uncomment here! Miguel: double check this!
            # =================================================================

            bg.append(self.integrator.y[-1] * 18)

            # self.num_iters += 5
            self.num_iters += self.n_solver_steps

        # Updating environment parameters
        self.simulation_state = self.integrator.y

        # Recording bg history for plotting and insulin for the state space
        self.bg_history = np.concatenate([self.bg_history, bg])
        self.insulin_history = np.concatenate([self.insulin_history, insulin_rate])

        # Miguel: What is this?
        # self.insulinOnBoard = np.zeros(1)
        # if self.bolusHistoryIndex > 0:
        #     for b in range(self.bolusHistoryIndex):
        #         self.insulinOnBoard = self.insulinOnBoard + self.bolusHistoryValue[b] * self.scalableExpIOB(self.num_iters - self.bolusHistoryTime[b], 75, 300)

        # Updating state
        self.state = np.concatenate([bg, list(reversed(self.insulin_history[-4:])), self.insulinOnBoard, bolus_given])

        done = 0

        #Set environment done = True if blood_glucose_level is negative, out of bounds or over the time limit
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

        # Miguel: que pasa?
        self.previous_action = action

        return np.array(self.state), np.mean(reward), done, {}


    def reset(self):
        ''' Basically a copy of the _init function

        '''

        # Reset bolus history
        self.bolusHistoryIndex = 0
        self.bolusHistoryValue = []
        self.bolusHistoryTime = []
        self.insulinOnBoard = np.zeros(1)

        # Reset sensor noise model -- Miguel: Make 
        # self.CGMerror = 0
        self.sensor_noise = np.random.rand(1)
        # self.CGMaux = []

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
        # initial_bg = 106
        # initial_insulin = np.zeros(4)
        initial_insulin = np.ones(4) * self.init_basal_optimal
        initial_iob = np.zeros(1)
        self.state = np.concatenate([np.repeat(initial_bg, self.stepsize), initial_insulin, initial_iob, np.zeros(1)])

        self.simulation_state = X0
        self.bg_history = []
        self.insulin_history = initial_insulin

        self.num_iters = 0


        # changing observation space if simulation time is changed -- This is slow!
        if self.simulation_time != 30:
            observation_space_shape = int(self.stepsize + 4 + 1)
            self.observation_space = spaces.Box(0, 500, (observation_space_shape,), dtype=np.float32)


        self.steps_beyond_done = None
        return np.array(self.state)


    def render(self, mode='human', close=False):
        #TODO: Clean up plotting routine

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
