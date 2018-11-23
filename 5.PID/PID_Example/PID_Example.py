import numpy as np
import sys

sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/Models')
sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/Modules')

from PID_Module import pid
from Linear_System_SPT import LinearSystem


def simulation():

    error = []

    model = LinearSystem(nsim=100, model_type='SISO', x0=np.array([0.5]), u0=np.array([1]), xs=np.array([1.5]),
                         us=np.array([3]), step_size=0.2)

    cost = 4.5

    for t in range(1, model.Nsim + 1):

        if t % 25 == 0:
            model.xs = np.array([2])

        if t % 50 == 0:
            model.xs = np.array([1])

        if t % 75 == 0:
            model.xs = np.array([1.5])

        # So it is consistent with RL
        if t < 5:
            inputs = np.array([1.])
        else:
            # kp = 1.5, ki = 1.8
            inputs = pid(model.xs, model.x[t - 1, :], model.x[t - 2, :], model.x[t - 2, :], kp=0.5, ki=1, kd=0,
                         u_1=model.u[t - 1, :], error=error, ts=1)

        next_state, _, _, _ = model.step(inputs, t, obj_function="MPC")

        cost += model.x[t] - model.xs

    return model, cost


if __name__ == "__main__":

    env, Cost = simulation()
