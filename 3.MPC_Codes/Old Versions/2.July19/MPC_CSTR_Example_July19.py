import mpctools as mpc
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/Models')

from CSTR_model_July19 import MimoCstr
from MPC_Module_July19 import ModelPredictiveControl


def simulation():

    # Model Initiation
    model = MimoCstr(nsim=50, k0=7.2e10)

    # MPC Object Initiation
    mpc_init = ModelPredictiveControl(model.Nsim, 10, model.Nx, model.Nu, 0.1, 0.1, 0.1, model.xs, model.us)

    # MPC Construction
    mpc_control = mpc_init.get_mpc_controller(model.cstr_ode_control, mpc_init.eval_time,
                                              np.concatenate([model.x0, [0, 0, 0]]), random_guess=False,
                                              verbosity=0)

    """
    Simulation portion
    """

    for t in range(model.Nsim):

        # Solve the MPC optimization problem
        mpc_init.solve_mpc(model.x, model.u, model.xsp, mpc_control, t, mpc_init.p)

        # Calculate the next stages
        model.x[t + 1, :] = model.next_state(model.x[t, :], model.u[t, :])

        # Update the P parameters for offset-free control
        mpc_init.p = model.x[t + 1, :] - mpc_init.x_predicted[t + 1, 0:3]

    print(model.cost_function())

    return model, mpc_init


if __name__ == "__main__":
    Model, MPC = simulation()
