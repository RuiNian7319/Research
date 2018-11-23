import mpctools as mpc
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/Models')

from CSTR_model import MimoCstr
from MPC_Module import ModelPredictiveControl


def simulation():

    # MPC Evaluation Period
    eval_period = 5

    # Model Initiation
    model = MimoCstr(nsim=50, k0=8.2e10)

    # MPC Initiation
    mpc_init = ModelPredictiveControl(10, model.Nx, model.Nu, 0.1, 0.1, 0.1, model.xs, model.us)

    # MPC Construction
    mpc_control = mpc_init.get_mpc_controller(model.cstr_ode, eval_period, model.x0, random=False, verbosity=0)

    # Output Disturbance
    output_disturb = np.zeros(model.Nx)
    x_corrected = np.zeros([model.Nsim + 1, model.Nx])

    """
    Simulation portion
    """

    for t in range(model.Nsim):

        """
        Disturbance
        """

        # if t == 10:
        #     model.disturbance()

        """
        MPC evaluation
        """

        if t % eval_period == 0 and t != 0:
            # Solve the MPC optimization problem
            mpc_init.solve_mpc(model.x, model.u, model.xsp, mpc_control, t)
            if t != 0:
                output_disturb = model.xs - model.x[t, :]
        elif t < 5:
            model.u[t, :] = [300, 0.1]
        else:
            model.u[t, :] = model.u[t - 1, :]

        # Calculate the next stages
        model.x[t + 1, :] = model.next_state(model.x[t, :], model.u[t, :])
        x_corrected[t + 1, :] = model.x[t + 1, :] + output_disturb

    print(model.cost_function(x_corrected))

    return model, mpc_init, x_corrected


if __name__ == "__main__":
    Model, MPC, x = simulation()
