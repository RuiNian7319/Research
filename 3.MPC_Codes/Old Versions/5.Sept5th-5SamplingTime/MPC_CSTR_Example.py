import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from CSTR_model_MPC import MimoCstr
from MPC_Module import ModelPredictiveControl


def simulation():

    # Plant Model
    model_plant = MimoCstr(nsim=314)

    # Build Controller Model
    model_control = MimoCstr(nsim=model_plant.Nsim, nx=model_plant.Nx * 2, xs=np.array([0.878, 324.5, 0.659, 0, 0, 0]),
                             x0=np.array([1, 310, 0.659, 0, 0, 0]), control=True)

    # MPC Object Initiation
    control = ModelPredictiveControl(model_control.Nsim, 5, model_control.Nx, model_control.Nu, 0.1, 0.1, 0,
                                     model_control.xs, model_control.us, dist=True, eval_time=5)

    # MPC Construction
    mpc_control = control.get_mpc_controller(model_control.cstr_ode, control.eval_time,
                                             model_control.x0, random_guess=False)

    """
    Simulation portion
    """

    for t in range(1, model_plant.Nsim + 1):

        # During evaluation periods
        if (t % control.eval_time == 0 or t == 1) and t != model_plant.Nsim:

            # Solve the MPC optimization problem, obtain current input and predicted state
            control_trajectory, state_trajectory = control.solve_mpc(model_plant.x, model_plant.xsp,
                                                                     mpc_control, t, control.p)
            for i in range(control.eval_time):
                model_control.u[t + i, :] = control_trajectory[i]
                model_control.x[t + i, :] = state_trajectory[i]

        # Calculate the next states for the plant
        model_plant.x[t, :] = model_plant.cstr_sim.sim(model_plant.x[t - 1, :], model_control.u[t, :])

        # Update the P parameters for offset-free control
        control.p = model_plant.x[t, :] - model_control.x[t, 0:3]

        # Copy control trajectories from control model to plant model
        if t == model_plant.Nsim:
            model_plant.u = model_control.u

    return model_plant, model_control, control


if __name__ == "__main__":
    Model_Plant, Model_Control, Controller = simulation()

    plt.plot(Model_Plant.x[:, 1])
    plt.show()
