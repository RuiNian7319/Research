import numpy as np
import matplotlib.pyplot as plt
import random

from Box import Box
from scipy.integrate import odeint
from copy import deepcopy


class SIMOSystem:

    def __init__(self, nsim, x0=np.array([2, 2/3]), u0=np.array([2]), xs=np.array([4, 4/3]), us=np.array([4]),
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
        self.Nx = 2
        self.Nu = 1
        self.action_space = Box(low=np.array([-5]), high=np.array([5]))
        self.observation_space = np.zeros(self.Nx)
        self.Q = q_cost * np.eye(self.Nx)
        self.R = r_cost
        self.S = s_cost

        self.A = np.array([[-2, 0], [0, -3]])
        self.B = np.array([2, 1])
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
        return "SIMOSystem({}, {}, {}, {}, {})".format(self.Nsim, self.x0, self.u0, self.xs, self.us, self.step_size)

    def __str__(self):
        return "SIMO system with {} time steps".format(self.Nsim)

    @staticmethod
    def seed(number):
        random.seed(number)
        np.random.seed(number)

    """
    Ordinary Differential Equation.  Solved using scipy.
    
    dxdt = -3x + u1 + u2
    """

    def ode(self, state, t, inputs):

        x1 = state[0]
        x2 = state[1]
        u = inputs[0]

        dxdt1 = self.A[0, 0] * x1 + self.B[0] * u
        dxdt2 = self.A[1, 1] * x2 + self.B[1] * u

        dxdt = [dxdt1, dxdt2]

        if self.control:
            dxdt.append(np.zeros(int(self.Nx * 0.5)))

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

            # Control input error on u1
            if 0.99 * self.xs[1] < self.x[step, 1] < 1.01 * self.xs[1]:
                reward += 1 - 0.35 * abs(self.x[step, 1] - self.xs[1])
            else:
                reward -= 0.35 * abs(self.x[step, 1] - self.xs[1])

            # Control input error on u2
            # if 0.99 * self.us[0] < self.u[step, 0] < 1.01 * self.us[0]:
            #     reward += 0.25 - 0.5 * abs(self.u[step, 0] - self.us[0])
            # else:
            #     reward -= 0.5 * abs(self.u[step, 0] - self.us[0])

            # Action penalty
            action = abs(self.u[step, 0] - self.u[step - 1, 0])
            if 0.9 * self.x[step - 1, 0] < self.x[step, 0] < 1.1 * self.x[step - 1, 0]:
                reward -= 0.15 * action
            else:
                reward -= 0.03 * action

            # Reward clipping
            reward = min(2, max(-2, reward))

        # MPC Reward function
        elif obj_function == "MPC":

            x = self.x[step, :] - self.xs
            u = (self.u[step] - self.us)[0]
            du = abs(self.u[step] - self.u[step - 1])[0]

            if delta_u == "l2":
                reward = - 0.77 * (x.T @ self.Q @ x + u * self.R * u + du * self.S * du) + 1
            elif delta_u == "l1":
                reward = - 0.77 * (x.T @ self.Q @ x + u * self.R * u + abs(du * self.S)) + 1
            else:
                reward = - 0.77 * (x.T @ self.Q @ x + u * self.R * u) + 1

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
        plt.title("State_1 Trajectories")
        plt.xlabel("Time, t (s)")
        plt.plot(self.x[:, 0])

        plt.subplot(1, 3, 2)
        plt.title("State_2 Trajectories")
        plt.xlabel("Time, t (s)")
        plt.plot(self.x[:, 1])

        plt.subplot(1, 3, 3)
        plt.title("Input Trajectories")
        plt.xlabel("Time, t (s)")
        plt.plot(self.u[:, 0])

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

    env = SIMOSystem(nsim=31)

    for time_step in range(1, env.Nsim + 1):

        if 30 < time_step < 60:
            control_input = np.array([4])
        elif 60 <= time_step:
            control_input = np.array([5])
        else:
            control_input = np.array([2])

        State, Reward, Done, Info = env.step(control_input, time_step, obj_function='MPC', delta_u="l2")
