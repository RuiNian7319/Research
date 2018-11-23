import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from CSTR_model_MPC import MimoCstr
from MPC_Module import ModelPredictiveControl


def simulation():

    # Plant Model
    model_plant = MimoCstr(nsim=39, x0=np.array([0.927, 317.78, 0.659]), u0=np.array([295, 0.1]))

    # Build Controller Model
    model_control = MimoCstr(nsim=model_plant.Nsim, nx=model_plant.Nx * 2, xs=np.array([0.878, 324.5, 0.659, 0, 0, 0]),
                             x0=np.array([0.927, 317.78, 0.659, 0, 0, 0]), u0=np.array([295, 0.1]), control=True,
                             k0=9.2e10)

    # MPC Object Initiation
    control = ModelPredictiveControl(model_control.Nsim, 10, model_control.Nx, model_control.Nu, 1, 0.5, 0.0,
                                     model_control.xs, model_control.us, dist=True)

    # MPC Construction
    mpc_control = control.get_mpc_controller(model_control.cstr_ode, control.eval_time,
                                             model_control.x0, random_guess=False)

    """
    Simulation portion
    """

    for t in range(1, model_plant.Nsim + 1):

        # Solve the MPC optimization problem, obtain current input and predicted state
        model_control.u[t, :], model_control.x[t, :] = control.solve_mpc(model_plant.x, model_plant.xsp,
                                                                         mpc_control, t, control.p)

        # Calculate the next states for the plant
        model_plant.x[t, :] = model_plant.next_state(model_plant.x[t - 1, :], model_control.u[t, :])

        # Disturbance
        if t % 20 == 0:
            model_plant.x[t, 1] -= 5

        # Update the P parameters for offset-free control
        control.p = model_plant.x[t, :] - model_control.x[t, 0:3]

        # Update the plant inputs to be same as controller inputs
        if t == model_plant.Nsim:
            model_plant.u = deepcopy(model_control.u)

    return model_plant, model_control, control


if __name__ == "__main__":
    Model_Plant, Model_Control, Controller = simulation()

    plt.plot(Model_Plant.x[:, 1])
    plt.show()
