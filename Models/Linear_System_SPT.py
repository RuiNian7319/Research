import numpy as np
import matplotlib.pyplot as plt
import random

from Box import Box
from scipy.integrate import odeint
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
                 us=np.array([10]), step_size=0.1, control=False, q_cost=1, r_cost=0.0, s_cost=0.1, random_seed=None):

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
            self.action_space = Box(low=-12, high=12, shape=(1,))
            self.Q = q_cost
            self.R = r_cost
            self.S = s_cost

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
            self.action_space = Box(low=np.array([-1.5, -0.8]), high=np.array([1.5, 0.8]))
            self.Q = q_cost * np.eye(self.Nx)
            self.R = r_cost * np.eye(self.Nu)
            self.S = s_cost * np.eye(self.Nu)

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

        # Seed the system for reproducability
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def __repr__(self):
        return "LinearSystem({}, {}, {})".format(self.Nsim, self.Nx, self.Nu)

    def __str__(self):
        return "{} System with {} time steps".format(self.model_type, self.Nsim)

    @staticmethod
    def seed(number):
        random.seed(number)
        np.random.seed(number)

    """
    Two ordinary differential equations (ODEs).  One SISO and one MIMO.
    t input is required for scipy to solve the integration.
    
    SISO ODE: dxdt = -4x + 2u
    
    MIMO ODE: dxdt1 = -3x1 - 2x2 + 4u1
              dxdt2 = -3x2 + 2u2
    """

    def ode(self, state, t, inputs):

        if self.model_type == 'SISO':
            x = state[0]
            u = inputs[0]

            dxdt = [float(self.A) * x + float(self.B) * u]

        elif self.model_type == 'MIMO':
            x1 = state[0]
            x2 = state[1]

            u1 = inputs[0][0]
            u2 = inputs[0][1]

            dxdt1 = self.A[0, 0] * x1 + self.A[0, 1] * x2 + self.B[0, 0] * u1 + self.B[0, 1] * u2
            dxdt2 = self.A[1, 0] * x1 + self.A[1, 1] * x2 + self.B[1, 0] * u1 + self.B[1, 1] * u2

            dxdt = [dxdt1, dxdt2]

        else:
            raise ValueError("Improper model specified")

        return dxdt

    """
    Takes one step in the simulation
    
    inputs: The action to the system
    time: The integration time [lower bound, upper bound]
    step: The simulation step; all integers
    """

    def step(self, inputs, step, obj_function="RL", delta_u="none"):

        x_next = odeint(self.ode, self.x[step - 1, :], [self.t[step - 1], self.t[step]], args=(inputs, ))

        self.x[step, :] = x_next[-1]
        self.u[step, :] = inputs[0]

        state = deepcopy(self.x[step, :])

        reward = self.reward_function(step, obj_function=obj_function, delta_u=delta_u)

        if step == (self.Nsim - 1):
            done = True
        else:
            done = False

        info = "placeholder"

        return state, reward, done, info

    """
    Reward function calculation
    """

    def reward_function(self, step, obj_function='RL', delta_u="none"):

        if obj_function == "RL":

            action_error = np.abs(np.sum(self.u[step, :] - self.us))

            reward = 0

            if self.model_type == "SISO":

                if 0.99 * self.xs[0] < self.x[step, 0] < 1.01 * self.xs[0]:
                    reward += 1 - 0.35 * abs(self.x[step, 0] - self.xs[0])
                else:
                    reward += - 0.35 * abs(self.x[step, 0] - self.xs[0])

                # Penalty for high actions
                reward -= 0.003 * action_error ** 2

                # Reward clipping for convergence.  Reward = [-1, 1]
                reward = min(1, max(reward, -1))

            elif self.model_type == "MIMO":

                # x1 reward
                if 0.98 * self.xs[0] < self.x[step, 0] < 1.02 * self.xs[0]:
                    reward += 0.5 - 0.1 * abs(self.x[step, 0] - self.xs[0])
                else:
                    reward += - 0.3 * abs(self.x[step, 0] - self.xs[0])

                # x2 reward
                if 0.98 * self.xs[1] < self.x[step, 1] < 1.02 * self.xs[1]:
                    reward += 0.5 - 0.1 * abs(self.x[step, 0] - self.xs[0])
                else:
                    reward += - 0.3 * abs(self.x[step, 1] - self.xs[1])

                # # u1 reward
                # if 0.98 * self.us[0] < self.u[step, 0] < 1.02 * self.us[0]:
                #     reward += 0.5 - 0.1 * abs(self.u[step, 0] - self.us[0])
                # else:
                #     reward += - 0.3 * abs(self.u[step, 0] - self.us[0])
                #
                # # u2 reward
                # if 0.98 * self.us[1] < self.u[step, 1] < 1.02 * self.us[1]:
                #     reward += 0.5 - 0.1 * abs(self.u[step, 1] - self.us[1])
                # else:
                #     reward += - 0.3 * abs(self.u[step, 1] - self.us[1])

                # Penalty for high actions
                # reward -= 0.005 * action**2
                if .9 * self.us[0] < self.u[step, 0] < 1.1 * self.us[0] and .9 * self.us[1] < self.u[step, 1] < 1.1 * self.us[1]:
                    reward -= 0.5 * action_error
                else:
                    reward -= 0.03 * action_error

                # Reward clipping for convergence.  Reward = [-1, 1]
                reward = min(1, max(reward, -1))

            else:
                raise ValueError("Improper model type specified.")

            """
            MPC Cost function as the reward
            """

        elif obj_function == "MPC":

            if self.model_type == "SISO":
                x = (self.x[step] - self.xs)[0]
                u = (self.u[step] - self.us)[0]
                du = (self.u[step] - self.u[step - 1])[0]

                if delta_u == "l2":
                    reward = - abs(x * self.Q * x + u * self.R * u + du * self.S * du)

                elif delta_u == "l1":
                    reward = - abs(x * self.Q * x + u * self.R * u + abs(du * self.S))

                else:
                    reward = - abs(x * self.Q * x + u * self.R * u)

                reward = max(-20, reward)

            elif self.model_type == "MIMO":
                x = self.x[step, :] - self.xs
                u = self.u[step, :] - self.us
                du = self.u[step, :] - self.u[step - 1, :]

                if delta_u == "l2":
                    reward = - (abs(x.T @ self.Q @ x + u.T @ self.R @ u + du.T @ self.S @ du))

                elif delta_u == "l1":
                    reward = - (abs(x.T @ self.Q @ x + u.T @ self.R @ u + sum(abs(du.T @ self.S))))

                else:
                    reward = - (abs(x.T @ self.Q @ x + u.T @ self.R @ u))

                reward = max(-20, reward)

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
            self.x = np.zeros([self.Nsim + 1, self.Nx])
            self.x[0, :] = self.x0
        else:
            self.x = np.zeros([self.Nsim + 1, self.Nx])
            self.x[0, :] = self.x0 + np.random.uniform(-2, 2)

        self.u = np.zeros([self.Nsim + 1, self.Nu])
        self.u[0, :] = self.u0

        return self.x[0, :]

    """
    Plots the state and input trajectories for the simple systems
    """

    def plots(self):

        if self.model_type == "SISO":
            plt.subplot(1, 2, 1)
            plt.title("State Trajectories")
            plt.xlabel("Time, t (s)")
            plt.plot(self.x[:])

            plt.subplot(1, 2, 2)
            plt.title("Input Trajectories")
            plt.xlabel("Time, t (s)")
            plt.plot(self.u[:])

            plt.show()

        else:
            plt.subplot(2, 2, 1)
            plt.title("State Trajectories")
            plt.plot(self.x[:, 0])

            plt.subplot(2, 2, 3)
            plt.xlabel("Time, t (s)")
            plt.plot(self.x[:, 1])

            plt.subplot(2, 2, 2)
            plt.title("Input Trajectories")
            plt.plot(self.u[:, 0])

            plt.subplot(2, 2, 4)
            plt.xlabel("Time, t (s)")
            plt.plot(self.u[:, 1])

            plt.show()

    """
    Controller performance assessment using integral of absolute error and integral of squared error

    Error: Input desired error type
    """

    def cost_function(self, x=None, error="MPC", transient_period=15):
        cost = 0

        if x is None:
            if error == "IAE":
                cost = sum(abs(self.x[1:] - self.xs)) / self.Nsim
            elif error == "ISE":
                cost = np.power(self.x[1:] - self.xs, 2)
                cost = sum(cost) / self.Nsim
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

            assert(x.shape[1] == len(self.xs))

            if error == "IAE":
                cost = sum(abs(x[1:] - self.xs)) / self.Nsim
            elif error == "ISE":
                cost = np.power(x[1:] - self.xs, 2)
                cost = sum(cost) / self.Nsim
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

    env = LinearSystem(nsim=31, model_type='MIMO', x0=np.array([0, 0]), u0=np.array([0, 0]),
                       xs=np.array([5, 4]), us=np.array([7, 3]), step_size=0.2)

    # env = LinearSystem(nsim=31, model_type='SISO', x0=np.array([0.5]), u0=np.array([1]), xs=np.array([5]),
    #                    us=np.array([10]), step_size=0.2)

    for time_step in range(1, env.Nsim + 1):

        if 30 < time_step < 60:
            control_input = np.array([[5, 4]])
        elif 60 <= time_step:
            control_input = np.array([[5, 7]])
        else:
            control_input = np.array([[0, 0]])

        # if 30 < time_step < 60:
        #     control_input = np.array([3])
        # elif 60 <= time_step:
        #     control_input = np.array([5])
        # else:
        #     control_input = np.array([0])

        State, Reward, Done, Info = env.step(control_input, time_step, obj_function='MPC', delta_u='l2')
