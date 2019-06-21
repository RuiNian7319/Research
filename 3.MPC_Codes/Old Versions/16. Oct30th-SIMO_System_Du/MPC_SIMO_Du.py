import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '/home/rui/Documents/Research/Models')
sys.path.insert(0, '/home/rui/Documents/Research/Modules')

from copy import deepcopy
from SIMO_CasADI import SIMOSystem
from MPC_Module_Discounted_du import ModelPredictiveControl


def simulation():

    # Plant Model
    model_plant = SIMOSystem(nsim=30, x0=np.array([2, 2/3]), u0=np.array([2]), xs=np.array([4, 4/3]), us=np.array([4]),
                             step_size=0.2, control=False, q_cost=1, r_cost=0.5, random_seed=1)

    # Build Controller Model
    model_control = SIMOSystem(nsim=model_plant.Nsim, x0=np.array([2., 2/3, 0, 0]), u0=np.array([2.]),
                               xs=np.array([4., 4/3, 0, 0]), us=np.array([4.]),
                               step_size=0.2, control=True, q_cost=1, r_cost=0.5, random_seed=1)

    # MPC Object Initiation
    control = ModelPredictiveControl(model_control.Nsim, 30, model_control.Nx, model_control.Nu, 1, 0.5, 0.0,
                                     model_control.xs, model_control.us, eval_time=model_plant.step_size, dist=True,
                                     gamma=0.9, s=0.2)

    """
    Simulation portion
    """

    for t in range(1, model_plant.Nsim + 1):

        # MPC Construction
        mpc_control = control.get_mpc_controller(model_control.system_ode, delta=control.eval_time, x0=model_control.x0,
                                                 random_guess=False, uprev=model_control.u[t - 1])

        # Solve the MPC optimization problem, obtain current input and predicted state
        model_control.u[t, :], model_control.x[t, :] = control.solve_mpc(model_plant.x, model_plant.xsp,
                                                                         mpc_control, t, control.p)

        # Calculate the next states for the plant
        model_plant.x[t, :] = model_plant.system_sim.sim(model_plant.x[t - 1, :], model_control.u[t, :])

        # Disturbance
        # if t % 20 == 0:
        #     model_plant.x[t, 1] -= 5

        # Update the P parameters for offset-free control
        control.p = model_plant.x[t, :] - model_control.x[t, 0:model_plant.Nx]

        # Update the plant inputs to be same as controller inputs
        if t == model_plant.Nsim:
            model_plant.u = deepcopy(model_control.u)

    return model_plant, model_control, control


if __name__ == "__main__":
    Model_Plant, Model_Control, Controller = simulation()

    # plt.plot(Model_Plant.x[:, 1])
    # plt.show()
