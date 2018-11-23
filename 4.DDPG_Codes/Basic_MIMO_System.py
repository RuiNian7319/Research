import numpy as np
from Box import Box
from copy import deepcopy


class BasicSystem:

    def __init__(self, nx=2, nu=2, nsim=100):
        self.Nx = nx
        self.Nu = nu
        self.Nsim = nsim
        self.observation_space = np.zeros(nx)
        self.action_space = Box(low=np.array([-2, -1]), high=np.array([2, 1]))
        self.x = np.zeros([self.Nsim, self.Nx])
        self.u = np.zeros([self.Nsim, self.Nu])
        self.xs = [100, 500]
        self.x[0, :] = [98, 503]

    def step(self, action, time):
        self.u[time, :] = action

        self.x[time, :] = self.x[time - 1, :] + self.u[time, :]

        reward = self.reward_function(time, action)

        if time == self.Nsim - 1:
            done = True
        else:
            done = False

        state = deepcopy(self.x[time, :])

        return state, reward, done

    def reset(self):
        self.x = np.zeros([self.Nsim, self.Nx])
        self.u = np.zeros([self.Nsim, self.Nu])

        self.x[0, :] = [98, 503]

    def reward_function(self, time, action):

        action = np.sum(action)
        reward = 0

        if 0.998 * self.xs[0] < self.x[time, 0] < 1.002 * self.xs[0]:
            reward += 15 - (abs(self.x[time, 0] - self.xs[0]) * 100)

        if 0.998 * self.xs[1] < self.x[time, 1] < 1.002 * self.xs[1]:
            reward += 15 - (abs(self.x[time, 0] - self.xs[0]) * 100)

        if 0.998 * self.xs[0] < self.x[time, 0] < 1.002 * self.xs[0]:
            reward += 15 - (abs(self.x[time, 0] - self.xs[0]) * 100)

        reward -= action ** 2

        reward = max(-50, reward)

        return reward


if __name__ == "__main__":
    env = BasicSystem()

    for t in range(1, env.Nsim):
        print(t)
        env.step(env.action_space.sample(), t)



