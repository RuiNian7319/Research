import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/Models')
sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/Modules')

from copy import deepcopy
from Linear_System_CasADI import LinearSystem
from MPC_Module_Discounted import ModelPredictiveControl


def simulation():

    # Plant Model
    model_plant = LinearSystem(nsim=100, model_type='SISO', x0=np.array([0.5]), u0=np.array([1]), xs=np.array([5]),
                               us=np.array([10]), step_size=0.2)

    # Build Controller Model
    model_control = LinearSystem(nsim=model_plant.Nsim, model_type=model_plant.model_type, x0=np.array([0.5, 0]),
                                 u0=np.array([1]), xs=np.array([5, 0]), us=np.array([10]), control=True, step_size=0.2)

    # MPC Object Initiation
    control = ModelPredictiveControl(model_control.Nsim, 10, model_control.Nx, model_control.Nu, 1, 0.5,
                                     0.0, model_control.xs, model_control.us, eval_time=model_plant.step_size,
                                     dist=True, gamma=0.9)

    # MPC Construction
    mpc_control = control.get_mpc_controller(model_control.system_ode, delta=control.eval_time,
                                             x0=model_control.x0, verbosity=0, random_guess=False)

    """
    Simulation portion
    """

    for t in range(1, model_plant.Nsim + 1):

        # Setpoint Change
        if t % 51 == 0:
            # MPC Object Initiation
            control = ModelPredictiveControl(49, 10, model_control.Nx, model_control.Nu, 1, 0.5,
                                             0.0, np.array([2.5]), np.array([5]), eval_time=model_plant.step_size,
                                             dist=True, gamma=0.9)

            # MPC Construction
            mpc_control = control.get_mpc_controller(model_control.system_ode, delta=control.eval_time,
                                                     x0=model_control.x0, verbosity=0, random_guess=False)

        # Solve the MPC optimization problem, obtain current input and predicted state
        model_control.u[t, :], model_control.x[t, :] = control.solve_mpc(model_plant.x, model_plant.xsp,
                                                                         mpc_control, t, control.p)

        # Calculate the next states for the plant
        model_plant.x[t, :] = model_plant.system_sim.sim(model_plant.x[t - 1, :], model_control.u[t, :])

        # Update the P parameters for offset-free control
        control.p = model_plant.x[t, :] - model_control.x[t, 0:model_plant.Nx]

        # Update the plant inputs to be same as controller inputs
        if t == model_plant.Nsim:
            model_plant.u = deepcopy(model_control.u)

    return model_plant, model_control, control


if __name__ == "__main__":
    Model_Plant, Model_Control, Controller = simulation()

    # Tests for validation
    assert(np.allclose(Model_Plant.x, Model_Control.x[:, 0:Model_Plant.Nx]))
