import mpctools as mpc
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, '/home/rui/Documents/Research/Models')

from CSTR_model import MimoCstr
from MPC_Module import ModelPredictiveControl


def simulation():

    # Plant Model
    model_plant = MimoCstr(nsim=50)

    # Build Controller Model
    model_control = MimoCstr(nsim=model_plant.Nsim, nx=model_plant.Nx * 2, xs=np.array([0.878, 324.5, 0.659, 0, 0, 0]),
                             x0=np.array([1, 310, 0.659, 0, 0, 0]), control=True)

    # MPC Object Initiation
    control = ModelPredictiveControl(model_control.Nsim, 10, model_control.Nx, model_control.Nu, 0.1, 0.1, 0.1,
                                     model_control.xs, model_control.us, dist=True)

    # MPC Construction
    mpc_control = control.get_mpc_controller(model_control.cstr_ode, control.eval_time,
                                             model_control.x0, random_guess=False)

    """
    Simulation portion
    """

    for t in range(model_plant.Nsim):

        # Solve the MPC optimization problem, obtain current input and predicted state
        model_control.u[t, :], model_control.x[t + 1, :] = control.solve_mpc(model_plant.x, model_plant.xsp,
                                                                             mpc_control, t, control.p)

        # Calculate the next states for the plant
        model_plant.x[t + 1, :] = model_plant.next_state(model_plant.x[t, :], model_control.u[t, :])

        # Update the P parameters for offset-free control
        control.p = model_plant.x[t + 1, :] - model_control.x[t + 1, 0:3]

    print(model_plant.cost_function())

    return model_plant, model_control, control


if __name__ == "__main__":
    Model_Plant, Model_Control, Controller = simulation()
