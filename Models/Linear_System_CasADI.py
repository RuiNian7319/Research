import numpy as np
import matplotlib.pyplot as plt
import random
import mpctools as mpc

from Box import Box
from copy import deepcopy


class LinearSystem:

    """
    Nsim: Length of simulation
    x0: Initial state
    u0: Initial input
    xs: Steady-state state
    us: Steady-state input
    model_type: Either SISO or MIMO models
    step_size: Step size in simulation
    t: Physical time of the system (some multiple of the step size)
    Nx: Number of states
    Nu: Number of inputs
    x: State trajectories
    u: Input trajectories
    A: System matrix
    B: Input matrix
    C: Output matrix
    D: Feedforward matrix
    """

    def __init__(self, nsim, model_type='SISO', x0=np.array([0.5]), u0=np.array([1]), xs=np.array([5]),
                 us=np.array([10]), step_size=0.1, control=False, q_cost=1, r_cost=0.5, random_seed=1):

        self.Nsim = nsim
        self.x0 = x0
        self.u0 = u0
        self.xs = xs
        self.us = us
        self.model_type = model_type
        self.step_size = step_size
        self.t = np.linspace(0, nsim*self.step_size, nsim + 1)
        self.control = control

        # Initialization Characteristics
        if model_type == "SISO":
            if self.control is True:
                self.Nx = 2
            else:
                self.Nx = 1
            self.Nu = 1
            self.action_space = Box(low=-5, high=5, shape=(1,))
            self.Q = q_cost
            self.R = r_cost

            # Model parameters
            self.A = np.array([-4])
            self.B = np.array([2])
            self.C = np.array([1])
            self.D = 0

        elif model_type == "MIMO":
            if self.control is True:
                self.Nx = 4
            else:
                self.Nx = 2
            self.Nu = 2
            self.action_space = Box(low=np.array([-5, -5]), high=np.array([5, 5]))
            self.Q = q_cost * np.eye(self.Nx)
            self.R = r_cost * np.eye(self.Nu)

            # Model parameters
            self.A = np.array([[-3, -2], [0, -3]])
            self.B = np.array([[4, 0], [0, 2]])
            self.C = np.array([1, 1])
            self.D = 0

        else:
            raise ValueError("Model type not specified properly")

        self.observation_space = np.zeros(self.Nx)

        # State and input trajectories
        self.x = np.zeros([nsim + 1, self.Nx])
        self.u = np.zeros([nsim + 1, self.Nu])
        self.x[0, :] = x0
        self.u[0, :] = u0

        self.xsp = np.zeros([self.Nsim + 1, self.Nx])
        self.xsp[0, :] = self.xs
        self.usp = np.zeros([self.Nsim + 1, self.Nu])
        self.usp[0, :] = self.us

        # Build the CasaDI functions
        self.system_sim = mpc.DiscreteSimulator(self.ode, self.step_size, [self.Nx, self.Nu], ["x", "u"])
        self.system_ode = mpc.getCasadiFunc(self.ode, [self.Nx, self.Nu], ["x", "u"], funcname="odef")

        # Seed the system for reproducability
        random.seed(random_seed)
        np.random.seed(random_seed)

    def __repr__(self):
        return "LinearSystem({}, {}, {})".format(self.Nsim, self.Nx, self.Nu)

    def __str__(self):
        return "{} System with {} time steps".format(self.model_type, self.Nsim)

    @staticmethod
    def seed(number):
        random.seed(number)

    """
    Two ordinary differential equations (ODEs).  One SISO and one MIMO.
    t input is required for scipy to solve the integration.
    
    SISO ODE: dxdt = -4x + u2
    
    MIMO ODE: dxdt1 = -3x1 - 2x2 + 4u1
              dxdt2 = -3x2 + 2u2
    """

    def ode(self, state, inputs):

        if self.model_type == 'SISO':
            x = state[0]
            u = inputs[0]

            dxdt = [float(self.A) * x + float(self.B) * u]

            if self.control:
                dxdt.append(np.zeros(int(self.Nx * 0.5)))

        elif self.model_type == 'MIMO':
            x1 = state[0]
            x2 = state[1]

            u1 = inputs[0]
            u2 = inputs[1]

            dxdt1 = self.A[0, 0] * x1 + self.A[0, 1] * x2 + self.B[0, 0] * u1 + self.B[0, 1] * u2
            dxdt2 = self.A[1, 0] * x1 + self.A[1, 1] * x2 + self.B[1, 0] * u1 + self.B[1, 1] * u2

            dxdt = [dxdt1, dxdt2]

            if self.control:
                dxdt.append(np.zeros(int(self.Nx * 0.5)))

        else:
            raise ValueError("Improper model specified")

        return dxdt

    """
    Takes one step in the simulation
    
    inputs: The action to the system
    time: The integration time [lower bound, upper bound]
    step: The simulation step; all integers
    """

    def step(self, inputs, step, obj_function="RL"):

        x_next = self.system_sim.sim(self.x[step - 1, :], inputs)

        self.x[step, :] = x_next[-1]
        self.u[step, :] = inputs

        state = deepcopy(x_next)

        reward = self.reward_function(step, obj_function=obj_function)

        if step == (self.Nsim - 1):
            done = True
        else:
            done = False

        info = "placeholder"

        return state, reward, done, info

    """
    Reward function calculation
    """

    def reward_function(self, step, obj_function='RL'):

        if obj_function == "RL":

            action = np.sum(self.u[step, :] - self.u[step - 1, :])

            reward = 0

            if self.model_type == "SISO":

                if 0.99 * self.xs < self.x[step, 0] < 1.01 * self.xs[0]:
                    reward += 15 - abs(self.x[step, 0] - self.xs[0])
                else:
                    reward += - abs(self.x[step, 0] - self.xs[0])

            elif self.model_type == "MIMO":

                if 0.99 * self.xs[0] < self.x[step, 0] < 1.01 * self.xs[0]:
                    reward += 15 - abs(self.x[step, 0] - self.xs[0])
                else:
                    reward += - abs(self.x[step, 0] - self.xs[0])

                if 0.99 * self.xs[1] < self.x[step, 1] < 1.01 * self.xs[1]:
                    reward += 15 - abs(self.x[step, 0] - self.xs[0])
                else:
                    reward += - abs(self.x[step, 1] - self.xs[1])

            else:
                raise ValueError("Improper model type specified.")

            reward -= (0.005 * action ** 2)

            """
            MPC Cost function as the reward
            """

        elif obj_function == "MPC":

            if self.model_type == "SISO":
                x = self.x[step] - self.xs
                u = self.u[step] - self.us

                reward = x * self.Q * x + u * self.R * u

            elif self.model_type == "MIMO":
                x = self.x[step, :] - self.xs
                u = self.u[step, :] - self.us

                reward = x.T @ self.Q @ x + u.T @ self.R @ u

            else:
                raise ValueError("Improper model type specified")

        else:
            raise ValueError("Improper reward function specified")

        return reward

    """
    Resets the state and input trajectories
    """

    def reset(self, random_init=False):
        if random_init is False:
            self.x = np.zeros([self.Nsim, self.Nx])
            self.x[0, :] = self.x0
        else:
            self.x = np.zeros([self.Nsim, self.Nx])
            self.x[0, :] = self.x0 + np.random.uniform(-2, 2)

        self.u = np.zeros([self.Nsim, self.Nu])
        self.u[0, :] = self.u0

        return self.x[0, :]

    """
    Plots the state and input trajectories for the simple systems
    """

    def plots(self):

        if self.model_type == "SISO":
            plt.subplot(1, 2, 1)
            plt.legend("State Trajectories")
            plt.plot(self.x[:])

            plt.subplot(1, 2, 2)
            plt.legend("Input trajectories")
            plt.plot(self.u[:])
            plt.show()

        else:
            plt.subplot(2, 2, 1)
            plt.legend("State Trajectories")
            plt.plot(self.x[:, 0])

            plt.subplot(2, 2, 3)
            plt.plot(self.x[:, 1])

            plt.subplot(2, 2, 2)
            plt.legend("Input Trajectories")
            plt.plot(self.u[:, 0])

            plt.subplot(2, 2, 4)
            plt.plot(self.u[:, 1])

            plt.show()

    """
    Controller performance assessment using integral of absolute error and integral of squared error

    Error: Input desired error type
    """

    def cost_function(self, x=None, error="MPC", transient_period=15):
        cost = 0

        """
        Evaluate for current system
        """

        if x is None:

            # Integral of absolute error
            if error == "IAE":
                cost = sum(abs(self.x[1:] - self.xs)) / self.Nsim

            # Integral of squared error
            elif error == "ISE":
                cost = np.power(self.x[1:] - self.xs, 2)
                cost = sum(cost) / self.Nsim

            # MPC loss function
            elif error == "MPC":
                if self.control is True:
                    nx = self.Nx / 2
                else:
                    nx = self.Nx
                dx = np.subtract(self.x[1:transient_period, 0:nx], self.xs[0:nx])

                q = self.Q * np.eye(nx)

                du = np.subtract(self.u[1:transient_period, 0:self.Nu], self.us)
                r = self.R * np.eye(self.Nu)

                for i in range(dx.shape[0]):
                    cost += dx[i, :] @ q @ dx[i, :].T + du[i, :] @ r @ du[i, :].T
            else:
                print("Error 1: Unspecified Loss Type")

        else:

            """
            Evaluate for a random system
            """

            assert(x.shape[1] == len(self.xs))

            # Integral of absolute error
            if error == "IAE":
                cost = sum(abs(x[1:] - self.xs)) / self.Nsim

            # Integral of squared error
            elif error == "ISE":
                cost = np.power(x[1:] - self.xs, 2)
                cost = sum(cost) / self.Nsim

            # MPC loss function
            elif error == "MPC":
                if self.control is True:
                    nx = self.Nx / 2
                else:
                    nx = self.Nx
                dx = np.subtract(self.x[1:transient_period, 0:nx], self.xs[0:nx])

                q = self.Q * np.eye(nx)

                du = np.subtract(self.u[1:transient_period, 0:self.Nu], self.us)
                r = self.R * np.eye(self.Nu)

                for i in range(dx.shape[0]):
                    cost += dx[i, :] @ q @ dx[i, :].T + du[i, :] @ r @ du[i, :].T
            else:
                print("Error 1: Unspecified Loss Type")

        return cost


if __name__ == "__main__":

    env = LinearSystem(nsim=100, model_type='MIMO', x0=np.array([0, 0]), u0=np.array([0, 0]),
                       xs=np.array([5, 4]), us=np.array([7, 3]), step_size=0.1)

    # env = LinearSystem(nsim=100, model_type='SISO', x0=np.array([0.5]), u0=np.array([1]), xs=np.array([5]),
    #                    us=np.array([10]), step_size=0.5)

    for time_step in range(1, env.Nsim):

        if 30 < time_step < 60:
            control_input = np.array([[3, 6]])
        elif 60 <= time_step:
            control_input = np.array([[5, 7]])
        else:
            control_input = np.array([[1, 3]])

        State, Reward, Done, Info = env.step(control_input, time_step)
