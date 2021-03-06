"""
Model Predictive Control code

Rui Nian
Last Updated: 03-July-2018
"""

import mpctools as mpc
import numpy as np


class ModelPredictiveControl:

    """
    Nt = Prediction and Control Horizon
    Nx = Number of states in the system
    Nu = Number of inputs into the system
    Q = State tuning matrix
    R = Input tuning matrix
    P = Terminal cost tuning matrix
    loss = Loss function for the MPC
    Pf = Terminal cost for the MPC

    upp_const = Upper bound for constraints
    low_const = Lower bound for constraints
    """

    def __init__(self, nsim, nt, num_of_states, num_of_inputs, q, r, p, ss_states, ss_inputs, eval_time=1, dist=False,
                 gamma=0.9, upp_u_const=None, low_u_const=None, upp_x_const=None, low_x_const=None):
        self.Nsim = nsim
        self.Nt = nt
        self.Nx = num_of_states
        self.Nu = num_of_inputs
        # self.Q = q * np.eye(int(num_of_states / 4))
        self.Q = q * np.eye(1)
        self.R = r * np.eye(num_of_inputs)
        self.P = p * np.eye(int(num_of_states / 4))
        self.ss_states = ss_states
        self.ss_inputs = ss_inputs
        self.eval_time = eval_time
        self.p = np.zeros(int(self.Nx * 0.5))
        self.dist = dist

        self.upp_u_const = upp_u_const
        self.low_u_const = low_u_const

        self.upp_x_const = upp_x_const
        self.low_x_const = low_x_const

        # Discount the stage costs.
        self.parameter = np.linspace(0, nt - 1, nt)
        self.gamma = gamma

        # If there is output disturbance, the Q and P matrices should only be half the size because the other states are
        # for the output disturbance modelling

        if self.dist is True:
            # self.Q = q * np.eye(int(num_of_states * 0.5))
            self.Q = q * np.eye(1)
            self.P = p * np.eye(int(num_of_states * 0.5))
        else:
            self.Q = q * np.eye(num_of_states)
            self.P = p * np.eye(num_of_states)

    def __repr__(self):
        return "ModelPredictiveControl({}, {}, {}, {})".format(self.Nsim, self.Nt, self.Nx, self.Nu)

    def __str__(self):
        return "This is a MPC controller with {} prediction and control horizon".format(self.Nt)

    """
    Calculation for each stages' cost
    
    states = Plant current states
    inputs = Current inputs into the plant
    
    Returns stage loss function of state and input
    """

    def stage_loss(self, states, inputs, parameter):

        # First part is actual state, 2nd part is offset free control
        dx = states[:int(self.Nx * 0.5)] + states[int(self.Nx * 0.5):self.Nx] - self.ss_states[:int(self.Nx*0.5)]
        du = inputs - self.ss_inputs

        x_cost = mpc.mtimes(dx.T, self.Q, dx)
        u_cost = mpc.mtimes(du.T, self.R, du)

        return (x_cost + u_cost)*self.gamma**parameter

    """
    Calculates the output loss
    
    """

    def output_loss(self, states, inputs, parameter):

        x = states[:int(self.Nx * 0.5)] + states[int(self.Nx * 0.5):self.Nx]
        # y = np.array([(0.776 * x[0] - 0.9 * x[2]), (0.6055 * x[1] - 1.3472 * x[3])]) - np.array([100, 0])
        y = np.array([(0.776 * x[0] - 0.9 * x[2]), ]) - np.array([100])

        du = inputs - self.ss_inputs

        y_cost = mpc.mtimes(y.T, self.Q, y)
        u_cost = mpc.mtimes(du.T, self.R, du)

        return (y_cost + u_cost)*self.gamma**parameter

    """
    Calculation for terminal cost
    
    states = Plant current states
    
    Returns terminal constraint that is a function of state
    """

    def terminal_loss(self, states):

        # First term is real states, 2nd term is the output disturbance correction.
        dx = states[:int(self.Nx * 0.5)] + states[int(self.Nx * 0.5):self.Nx] - self.ss_states[:int(self.Nx*0.5)]

        return mpc.mtimes(dx.T, self.P, dx)

    """
    MPC Controller construction
    
    ode_casadi = The discrete simulator from the model
    delta = Model split intervals
    x0 = Initial states
    random = If true, the initial guesses are random, if false, the guesses are the steady state values
    c = Collocation points
    verbosity = Amount of information returned by MPCTools
    upper_bound = The upper bound of the states and inputs
    lower_bound = The lower bound of the states and inputs
    
    mpc_controller = Creates the MPC controller
    """

    def get_mpc_controller(self, ode_casadi, delta, x0, random_guess=False, c=3, verbosity=0, upper_bound=1.8,
                           lower_bound=0.2):

        # A very sketchy way to find the amount of time varying parameters
        if self.parameter.shape == (self.parameter.shape[0], ):
            shape = 1
        else:
            shape = self.parameter.shape[1]

        loss = mpc.getCasadiFunc(self.output_loss, [self.Nx, self.Nu, shape], ["x", "u", "p"], funcname="loss")
        # pf = mpc.getCasadiFunc(self.terminal_loss, [self.Nx], ["x"], funcname="pf")

        # Define where the function args should be.  More precisely, we must put p in the loss function.
        funcargs = {"f": ["x", "u"], "l": ["x", "u", "p"]}

        if random_guess is True:
            guess = {"x": np.random.uniform(5, 10, self.Nx), "u": np.random.uniform(5, 10, self.Nu)}
        else:
            guess = {"x": self.ss_states, "u": self.ss_inputs}

        # Set "c" above 1 to use collocation
        contargs = dict(
            N={"t": self.Nt, "x": self.Nx, "u": self.Nu, "c": c, "p": 1},
            verbosity=verbosity,
            l=loss,
            x0=x0,
            p=self.parameter,
            # Upper and Lower State Constraints
            ub={"u": self.upp_u_const,
                "x": self.upp_x_const},
            lb={"u": self.low_u_const,
                "x": self.low_x_const},
            guess=guess,
            funcargs=funcargs
        )

        # Create MPC controller
        mpc_controller = mpc.nmpc(f=ode_casadi, Delta=delta, **contargs)

        return mpc_controller

    """
    Solves the MPC optimization problem
    """

    def solve_mpc(self, model_states, model_sp, mpc_controller, sim_time, p):

        # Fix the initial x, so optimization cannot change the current x
        if self.dist is True:
            mpc_controller.fixvar("x", 0, np.concatenate([model_states[sim_time - 1, :], p]))
        else:
            mpc_controller.fixvar("x", 0, model_states[sim_time - 1, :])

        # Solve MPC
        mpc_controller.solve()

        # Ensure the MPC solved the optimization problem
        if mpc_controller.stats["status"] != "Solve_Succeeded":
            print("MPC did not solve at time step {}".format(sim_time))
        else:
            mpc_controller.saveguess()

        # Input the current input into the plant, discard the rest
        model_input = (np.squeeze(mpc_controller.var["u", 0]))

        # Find the predicted x value from the controller
        x_predicted = np.squeeze(mpc_controller.var["x", 1])

        # Model set point.  Different dimensions depending on if output disturbance is true or not
        if self.dist is True:
            model_sp[sim_time, :] = self.ss_states[:int(self.Nx * 0.5)]
        else:
            model_sp[sim_time, :] = self.ss_states[:self.Nx]

        return model_input, x_predicted
