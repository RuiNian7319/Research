import mpctools as mpc
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/Models')

from RL_Module import *
from CSTR_model import *

"""
Define reward function for RL.  User defines the reward function structure.  The below is an example.
"""


def simulation():

    # Model Initiation
    model = MimoCstr(nsim=315, k0=7.2e10)

    # Reinforcement Learning Initiation
    rl = ReinforceLearning(discount_factor=0.95, states_start=300, states_stop=340, states_interval=0.5,
                           actions_start=-15, actions_stop=15, actions_interval=2.5, learning_rate=0.5,
                           epsilon=0.2, doe=1.2, eval_period=1)

    """
    Example of user defined states and actions.  Users do not need to do this.  This is only if users want to define 
    their own states and actions.  RL will automatically populate states and actions if user does not input their own.
    """

    states = np.zeros([50])
    states[0:10] = np.linspace(290, 310, 10)
    states[10:40] = np.linspace(310, 330, 30)
    states[40:50] = np.linspace(330, 350, 10)

    rl.user_states(list(states))

    actions = np.zeros([36])
    actions[0:5] = np.linspace(290, 298, 5)
    actions[5:30] = np.linspace(298, 302, 25)
    actions[30:35] = np.linspace(302, 310, 5)

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

    rlist = []

    for episode in range(1):

        # Reset the model after each episode
        model.reset(random_init=False)
        tot_reward = 0
        state = 0
        action = 0

        for t in range(1, model.Nsim + 1):

            """
            Disturbance
            """

            if t % 50 == 0:
                model.x[t - 1, 1] += 5

            """
            RL Evaluate
            """

            if t % rl.eval_period == 0:
                state, action = rl.ucb_action_selection(model.x[t - 1, 1])
                model.u[t, 0], action = rl.action_selection(state, action, model.u[t-1, 0], 25, ep_greedy=False, time=t,
                                                            min_eps_rate=0.1)
            else:
                model.u[t, :] = model.u[t - 1, :]

            model.x[t, :] = model.next_state(model.x[t - 1, :], model.u[t, :])

            """
            Feedback evaluation
            """

            if t == rl.eval_feedback:
                reward = model.reward_function(t)
                rl.matrix_update(action, reward, state, model.x[t, 1], 5)
                tot_reward = tot_reward + reward

        rlist.append(tot_reward)

        rl.autosave(episode, 250)

        if episode % 100 == 0:
            print(model.cost_function())

    return model, rl, rlist


# if __name__ == "__main__":
Model, RL, rList = simulation()
