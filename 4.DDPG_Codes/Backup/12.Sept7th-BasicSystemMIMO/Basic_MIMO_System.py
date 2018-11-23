import numpy as np
from Box import Box
from copy import deepcopy


class BasicSystem:

    def __init__(self, nx=4, nu=3, nsim=200):
        self.Nx = nx
        self.Nu = nu
        self.Nsim = nsim
        self.observation_space = np.zeros(nx)
        self.action_space = Box(low=np.array([-2, -10, -2]), high=np.array([2, 10, 2]))
        self.x = np.zeros([self.Nsim, self.Nx])
        self.u = np.zeros([self.Nsim, self.Nu])
        self.xs = [100, 500, 10, 1]
        self.x[0, :] = [98, 503, 6, 1]

    @staticmethod
    def seed(number):
        np.random.seed(number)

    def step(self, action, time):
        self.u[time, :] = action

        self.x[time, 0:self.Nu] = self.x[time - 1, 0:self.Nu] + self.u[time, :]

        reward = self.reward_function(time, action)

        if time == self.Nsim - 1:
            done = True
        else:
            done = False

        state = deepcopy(self.x[time, :])

        info = "placeholder"

        return state, reward, done, info

    def reset(self):
        self.x = np.zeros([self.Nsim, self.Nx])
        self.u = np.zeros([self.Nsim, self.Nu])

        self.x[0, :] = [98, 503, 6, 1]

        return self.x[0, :]

    def reward_function(self, time, action):

        action = np.sum(action)
        reward = 0

        if 0.99 * self.xs[0] < self.x[time, 0] < 1.01 * self.xs[0]:
            reward += 15 - (abs(self.x[time, 0] - self.xs[0]) * 10)
        else:
            reward += -abs(self.x[time, 0] - self.xs[0])

        if 0.99 * self.xs[1] < self.x[time, 1] < 1.01 * self.xs[1]:
            reward += 15 - (abs(self.x[time, 1] - self.xs[1]))
        else:
            reward += -abs(self.x[time, 1] - self.xs[1])

        if 0.99 * self.xs[2] < self.x[time, 2] < 1.01 * self.xs[2]:
            reward += 15 - (abs(self.x[time, 2] - self.xs[2]) * 5)
        else:
            reward += -abs(self.x[time, 2] - self.xs[2])

        reward -= action ** 2

        reward = max(-100, reward)

        return reward


if __name__ == "__main__":
    env = BasicSystem()

    for t in range(1, env.Nsim):
        print(t)
        env.step(env.action_space.sample(), t)



