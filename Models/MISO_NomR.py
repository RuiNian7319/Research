import numpy as np
import matplotlib.pyplot as plt
import random

from Box import Box
from scipy.integrate import odeint
from copy import deepcopy


class MISOSystem:

    def __init__(self, nsim, x0=np.array([3]), u0=np.array([3, 6]), xs=np.array([4.5]), us=np.array([5.5, 8]),
                 step_size=0.2, control=False, q_cost=1, r_cost=0.5, s_cost=0.1, random_seed=1):
        self.Nsim = nsim
        self.x0 = x0
        self.u0 = u0
        self.xs = xs
        self.us = us
        self.step_size = step_size
        self.t = np.linspace(0, nsim * self.step_size, nsim + 1)
        self.control = control

        # Model Parameters
        self.Nx = 1
        self.Nu = 2
        self.action_space = Box(low=np.array([-2, -2]), high=np.array([2, 2]))
        self.observation_space = np.zeros(self.Nx)
        self.Q = q_cost
        self.R = r_cost * np.eye(self.Nu)
        self.S = s_cost * np.eye(self.Nu)

        self.A = np.array([-3])
        self.B = np.array([1, 1])
        self.C = np.array([1])
        self.D = 0

        # State and Input Trajectories
        self.x = np.zeros([nsim + 1, self.Nx])
        self.u = np.zeros([nsim + 1, self.Nu])
        self.x[0, :] = x0
        self.u[0, :] = u0

        # Seed the system for reproducability
        random.seed(random_seed)
        np.random.seed(random_seed)

    def __repr__(self):
        return "MISOSystem({}, {}, {}, {}, {})".format(self.Nsim, self.x0, self.u0, self.xs, self.us, self.step_size)

    def __str__(self):
        return "MISO system with {} time steps".format(self.Nsim)

    @staticmethod
    def seed(number):
        random.seed(number)
        np.random.seed(number)

    """
    Ordinary Differential Equation.  Solved using scipy.
    
    dxdt = -3x + u1 + u2
    """

    def ode(self, state, t, inputs):

        x = state[0]
        u1 = inputs[0][0]
        u2 = inputs[0][1]

        dxdt = self.A * x + self.B[0] * u1 + self.B[1] * u2

        return dxdt

    """
    Simulates one step
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
    Reward function for the system
    """

    def reward_function(self, step, obj_function='RL', delta_u="none"):

        # RL Reward function
        if obj_function == "RL":

            reward = 0

            # Set-point tracking error
            if 0.99 * self.xs[0] < self.x[step, 0] < 1.01 * self.xs[0]:
                reward += 1 - 0.35 * abs(self.x[step, 0] - self.xs[0])
            else:
                reward -= 0.35 * abs(self.x[step, 0] - self.xs[0])

            # # Control input error on u1
            # if 0.99 * self.us[0] < self.u[step, 0] < 1.01 * self.us[0]:
            #     reward += 0.25 - 0.35 * (self.u[step, 0] - self.us[0])
            # else:
            #     reward -= 0.35 * (self.u[step, 0] - self.us[0])
            #
            # # Control input error on u2
            # if 0.99 * self.us[1] < self.u[step, 1] < 1.01 * self.us[1]:
            #     reward += 0.25 - 0.5 * (self.u[step, 1] - self.us[1])
            # else:
            #     reward -= 0.5 * (self.u[step, 1] - self.us[1])

            # Reward clipping
            reward = min(1.25, max(-1.25, reward))

        # MPC Reward function
        elif obj_function == "MPC":

            x = (self.x[step] - self.xs)[0]
            u = self.u[step, :] - self.us
            du = abs(self.u[step - 1, :] - self.u[step, :])

            if delta_u == "l2":
                reward = - 0.8 * (abs(x * self.Q * x + u.T @ self.R @ u + du.T @ self.S @ du)) + 1

            elif delta_u == "l1":
                reward = - 0.8 * (abs(x * self.Q * x + u.T @ self.R @ u + sum(abs(du @ self.S)))) + 1

            else:
                reward = - 0.8 * (abs(x * self.Q * x + u.T @ self.R @ u)) + 1

            reward = max(-10, reward)

        # Improper Reward function
        else:
            raise ValueError("Improper model type specified.")

        return reward

    """
    Reset the simulation with a choice of initializing the parameters randomly.
    """

    def reset(self, random_init=False):
        if random_init is False:
            self.x = np.zeros([self.Nsim + 1, self.Nx])
            self.x[0] = self.x0
        else:
            self.x = np.zeros([self.Nsim + 1, self.Nx])
            self.x[0] = self.x0 + np.random.uniform(-2, 2)

        self.u = np.zeros([self.Nsim + 1, self.Nu])
        self.u[0, :] = self.u0

        return self.x[0, :]

    """
    Plots the state and input trajectories
    """

    def plots(self):
        plt.subplot(1, 3, 1)
        plt.title("State Trajectories")
        plt.xlabel("Time, t (s)")
        plt.plot(self.x[:])

        plt.subplot(1, 3, 2)
        plt.title("Input_1 Trajectories")
        plt.xlabel("Time, t (s)")
        plt.plot(self.u[:, 0])

        plt.subplot(1, 3, 3)
        plt.title("Input_2 Trajectories")
        plt.xlabel("Time, t (s)")
        plt.plot(self.u[:, 1])

        plt.show()

    """
    Controller performance assessment
    """

    def cost_function(self, error="MPC", transient_period=15):
        cost = 0

        if error == "MPC":
            nx = self.Nx
            dx = np.subtract(self.x[1:transient_period, 0:nx], self.xs[0:nx])

            q = self.Q * np.eye(nx)

            du = np.subtract(self.u[1:transient_period, 0:self.Nu], self.us)
            r = self.R * np.eye(self.Nu)

            for i in range(dx.shape[0]):
                cost += dx[i, :] @ q @ dx[i, :].T + du[i, :] @ r @ du[i, :].T

        else:
            raise ValueError("Undefined error type")

        return cost


if __name__ == "__main__":

    env = MISOSystem(nsim=31)

    for time_step in range(1, env.Nsim + 1):

        if 30 < time_step < 60:
            control_input = np.array([[5.5, 8]])
        elif 60 <= time_step:
            control_input = np.array([[5, 7]])
        else:
            control_input = np.array([[3, 6]])

        State, Reward, Done, Info = env.step(control_input, time_step, obj_function='MPC', delta_u="l1")
