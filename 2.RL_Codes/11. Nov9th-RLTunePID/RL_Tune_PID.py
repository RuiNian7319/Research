import numpy as np
import sys
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/Models')
sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/Modules')

from RL_Module import ReinforceLearning
from PID_Sim import sim
"""
Define reward function for RL.  User defines the reward function structure.  The below is an example.
"""


def simulation():

    # Reinforcement Learning Initiation
    rl = ReinforceLearning(discount_factor=0.95, states_start=300, states_stop=340, states_interval=0.5,
                           actions_start=-15, actions_stop=15, actions_interval=2.5, learning_rate=0.5,
                           epsilon=0.8, doe=0, eval_period=1)

    """
    Example of user defined states and actions.  Users do not need to do this.  This is only if users want to define 
    their own states and actions.  RL will automatically populate states and actions if user does not input their own.
    """

    states = []

    rl.x1 = np.linspace(1.5, 3.5, 16)
    rl.x2 = np.linspace(3.5, 5.5, 16)

    for i in rl.x1:
        for j in rl.x2:
            states.append([i, j])

    rl.user_states(list(states))

    actions = []

    rl.u1 = np.linspace(-0.4, 0.4, 5)
    rl.u2 = np.linspace(-0.4, 0.4, 5)

    for i in rl.u1:
        for j in rl.u2:
            actions.append([i, j])

    rl.user_actions(list(actions))

    """
    Load pre-trained Q, T and NT matrices
    """

    q = np.loadtxt("Q_Matrix.txt")
    t = np.loadtxt("T_Matrix.txt")
    nt = np.loadtxt("NT_Matrix.txt")

    rl.user_matrices(q, t, nt)

    """
    Simulation portion
    """

    sim_length = 90

    kp_ki = np.zeros([sim_length, 2])
    kp_ki[0, :] = np.array([1.4, 3.3])

    actions = np.zeros([sim_length, 2])

    for t in range(1, sim_length):

        """
        RL Evaluate
        """

        state, action = rl.ucb_action_selection(kp_ki[t - 1, :])
        action, action_index = rl.action_selection(state, action, actions[t - 1, :], no_decay=25,
                                                   ep_greedy=False, time=t,
                                                   min_eps_rate=0.4)
        actions[t, :] = action

        kp_ki[t, :], reward, x_trajectory = sim(kp_ki[t - 1], action)

        """
        Feedback evaluation
        """

        rl.matrix_update(action_index, reward, state, kp_ki[t, :], 5)

        # Bind actions
        if (kp_ki[t, :] > np.array([8, 8])).any() or (kp_ki[t, :] < np.array([0, 0])).any():
            kp_ki[t, :] = np.array([1.5, 3.5])

        rl.autosave(t, 250)

    return kp_ki, actions, rl


if __name__ == "__main__":

    Kp_Ki, Actions, RL = simulation()
