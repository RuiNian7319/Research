import numpy as np
import sys
from copy import deepcopy
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/Models')
sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/Modules')

from PID_Module import pid
from Linear_System_SPT import LinearSystem


def sim(state, action):

    # New Kp and Ki
    pid_params = state + action

    # PI Error
    error = []

    model = LinearSystem(nsim=40, model_type='SISO', x0=np.array([0.5]), u0=np.array([1]), xs=np.array([1.5]),
                         us=np.array([3]), step_size=0.2)

    reward = 3.5

    for t in range(1, model.Nsim + 1):

        if t % 10 == 0:
            model.xs = np.array([2])

        if t % 20 == 0:
            model.xs = np.array([1])

        if t % 30 == 0:
            model.xs = np.array([1.5])

        # kp = 1.5, ki = 1.8
        inputs = pid(model.xs, model.x[t - 1, :], model.x[t - 2, :], model.x[t - 2, :], kp=pid_params[0],
                     ki=pid_params[1], kd=0, u_1=model.u[t - 1, :], error=error, ts=1)

        next_state, _, _, _ = model.step(inputs, t, obj_function="MPC")

        reward -= abs(model.x[t] - model.xs)[0]
        reward = min(2, max(-5, reward))

    return pid_params, reward, model.x


if __name__ == "__main__":

    states = np.zeros((10, 2))
    states[0, :] = np.array([1.6, 3.3])

    for i in range(1, 10):

        if i == 2:
            Action = np.array([0.6, -0.4])
        else:
            Action = np.array([0, 0])

        states[i, :], Reward, Trajectory = sim(states[i - 1, :], Action)
        print(Reward)
