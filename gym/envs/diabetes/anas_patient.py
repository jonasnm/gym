"""
Anas El Fathis stable patients in an OpenAI gym environment

Inherits mostly from the HovorkaCambridge environment

"""

from gym.envs.diabetes import hovorka_cambridge

from gym.envs.diabetes.load_mcgill_patients import matlab_to_python

import numpy as np

# Gym imports
from gym import spaces
from gym.utils import seeding
from gym.envs.diabetes.meal_generator.meal_generator import meal_generator
from gym.envs.diabetes.hovorka_model import hovorka_model, hovorka_model_tuple

# ODE solver stuff
from scipy.integrate import ode
from scipy.optimize import fsolve

class AnasPatient(hovorka_cambridge.HovorkaCambridgeBase):

    def __init__(self, patient_number=0):
        """
        Initializing the simulation environment.
        """
        np.random.seed(1) ### Fixing seed


        self.previous_action = 0

        ## Loading variable parameters
        reward_flag, bg_init_flag = self._update_parameters()


        # Initialize bolus history
        self.bolusHistoryIndex = 0
        self.bolusHistoryValue = []
        self.bolusHistoryTime = []
        self.insulinOnBoard = np.zeros(1)

        # Initialize sensor model
        # self.CGMlambda = 15.96    # Johnson parameter of recalibrated and synchronized sensor error.
        # self.CGMepsilon = -5.471  # Johnson parameter of recalibrated and synchronized sensor error.
        # self.CGMdelta = 1.6898    # Johnson parameter of recalibrated and synchronized sensor error.
        # self.CGMgamma = -0.5444   # Johnson parameter of recalibrated and synchronized sensor error.
        self.CGMerror = 0
        self.sensor_noise = np.random.randn(1)
        # self.CGMaux = []
        self.sensorNoiseValue = 0.07 # Set a value

        # =======================
        # Anas patient parameters
        # =======================
        P, init_basal, carb_factor, _ = matlab_to_python(patient_number)
        init_basal_optimal = init_basal[0]
        self.P = P
        self.init_basal_optimal = init_basal_optimal
        self.bolus = carb_factor

        self.action_space = spaces.Box(0, 2*self.init_basal_optimal, (1,), dtype=np.float32)


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
        self.observation_space = spaces.Box(0, 500, (int(self.stepsize + 4 + 2),), dtype=np.float32)

        # State is BG, simulation_state is parameters of hovorka model
        initial_bg = X0[-1] * 18
        initial_insulin = np.ones(4) * self.init_basal_optimal
        initial_iob = np.zeros(1)
        self.state = np.concatenate([np.repeat(initial_bg, self.stepsize), initial_insulin, initial_iob, np.zeros(1)])

        self.simulation_state = X0

        # Keeping track of entire blood glucose level for each episode
        self.bg_history = []
        self.insulin_history = initial_insulin

        # ====================
        # Meal setup
        # ====================

        eating_time = 1
        meals, meal_indicator = meal_generator(eating_time=eating_time, premeal_bolus_time=0)

        self.meals = meals
        self.meal_indicator = meal_indicator
        self.eating_time = eating_time

        # Counter for number of iterations
        self.num_iters = 0

        # If blood glucose is less than zero, the simulator is out of bounds.
        self.bg_threshold_low = 0
        self.bg_threshold_high = 500

        # TODO: This number is arbitrary
        self.max_iter = 2160

        # Reward flag
        self.reward_flag = reward_flag

        self.steps_beyond_done = None


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
        if self.stepsize != 1:
            observation_space_shape = int(self.stepsize + 4 + 1)
            self.observation_space = spaces.Box(0, 500, (observation_space_shape,), dtype=np.float32)


        self.steps_beyond_done = None
        return np.array(self.state)