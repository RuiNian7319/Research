"""
Model Predictive Control code

Rui Nian
Last Updated: 03-July-2018
"""

import mpctools as mpc
import numpy as np
import matplotlib.pyplot as plt
from casadi import *


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
    """

    def __init__(self, nsim, nt, num_of_states, num_of_inputs, q, r, p, ss_states, ss_inputs, eval_time=1):
        self.Nsim = nsim
        self.Nt = nt
        self.Nx = num_of_states
        self.Nu = num_of_inputs
        self.Q = q * np.eye(num_of_states)
        self.R = r * np.eye(num_of_inputs)
        self.P = p * np.eye(num_of_states)
        self.ss_states = ss_states
        self.ss_inputs = ss_inputs
        self.eval_time = eval_time
        self.x_predicted = np.zeros([self.Nsim + 1, self.Nx * 2])
        self.p = np.zeros(self.Nx)

    def __repr__(self):
        return "ModelPredictiveControl({}, {}, {}, {})".format(self.x, self.y, self.z, self.a)

    def __str__(self):
        return "This is a MPC controller with {} prediction and control horizon".format(self.Nt)

    """
    Calculation for each stages' cost
    
    states = Plant current states
    inputs = Current inputs into the plant
    
    Returns stage loss function of state and input
    """

    def stage_loss(self, states, inputs):

        dx = states[:self.Nx] + states[self.Nx:self.Nx * 2] - self.ss_states
        du = inputs - self.ss_inputs

        return mpc.mtimes(dx.T, self.Q, dx) + mpc.mtimes(du.T, self.R, du)

    """
    Calculation for terminal cost
    
    states = Plant current states
    
    Returns terminal constraint that is a function of state
    """

    def terminal_loss(self, states):

        dx = states[:self.Nx] + states[self.Nx:self.Nx*2] - self.ss_states

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

    def get_mpc_controller(self, ode_casadi, delta, x0, random_guess=True, c=5, verbosity=0, upper_bound=1.2,
                           lower_bound=0.8):

        loss = mpc.getCasadiFunc(self.stage_loss, [self.Nx * 2, self.Nu], ["x", "u"], funcname="loss")
        pf = mpc.getCasadiFunc(self.terminal_loss, [self.Nx * 2], ["x"], funcname="pf")

        if random_guess is True:
            guess = {"x": np.random.uniform(5, 10, self.Nx), "u": np.random.uniform(5, 10, self.Nu)}
        else:
            guess = {"x": np.concatenate([self.ss_states, [0, 0, 0]]), "u": self.ss_inputs}

        # Set "c" above 1 to use collocation
        contargs = dict(
            N={"t": self.Nt, "x": self.Nx * 2, "u": self.Nu, "c": c},
            verbosity=verbosity,
            l=loss,
            x0=x0,
            Pf=pf,
            # Upper and Lower Confidence Bound
            ub={"u": upper_bound * self.ss_inputs, "x": upper_bound * np.concatenate([self.ss_states, [inf, inf, inf]])},
            lb={"u": lower_bound * self.ss_inputs, "x": lower_bound * np.concatenate([self.ss_states, [-inf, -inf, -inf]])},
            guess=guess
        )

        # Create MPC controller
        mpc_controller = mpc.nmpc(f=ode_casadi, Delta=delta, **contargs)

        return mpc_controller

    """
    Solves the MPC optimization problem
    """

    def solve_mpc(self, model_states, model_inputs, model_sp, mpc_controller, sim_time, p):

        # Fix the initial x, so optimization cannot change the current x
        mpc_controller.fixvar("x", 0, np.concatenate([model_states[sim_time, :], p]))

        # Solve MPC
        mpc_controller.solve()

        # Ensure the MPC solved the optimization problem
        if mpc_controller.stats["status"] != "Solve_Succeeded":
            print("MPC did not solve at time step {}".format(sim_time))
        else:
            mpc_controller.saveguess()

        # Input the current input into the plant, discard the rest
        model_inputs[sim_time, :] = np.squeeze(mpc_controller.var["u", 0])
        # Find the predicted x value from the controller
        self.x_predicted[sim_time + 1, :] = np.squeeze(mpc_controller.var["x", 1])
        model_sp[sim_time + 1, :] = self.ss_states
