import mpctools as mpc
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/Models')
sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/RL_Codes')
sys.path.insert(0, '/Users/ruinian/Documents/Research/2.RL_Codes')

from RL_Module import *

"""
Define reward function for RL.  User defines the reward function structure.  The below is an example.
"""


# Model for MechaTronix
def model(action):
    return 0.0104 * np.square(action) + 0.183 * action - 4.558


# Reward function
def reward_calc(state, setpoint):
    return max(-np.square(state - setpoint), -150)


def simulation():

    # Reinforcement Learning Initiation
    rl = ReinforceLearning(discount_factor=0.95, states_start=300, states_stop=340, states_interval=0.5,
                           actions_start=-15, actions_stop=15, actions_interval=2.5, learning_rate=0.5,
                           epsilon=0.2, doe=1.2, eval_period=7)

    """
    Example of user defined states and actions.  Users do not need to do this.  This is only if users want to define 
    their own states and actions.  RL will automatically populate states and actions if user does not input their own.
    """

    states = np.linspace(-2, 35, 38)
    rl.user_states(list(states))

    actions = np.linspace(0, 60, 61)
    rl.user_actions(list(actions))

    # states = np.linspace(-10, 10, 21)
    # rl.user_states(list(states))
    #
    # actions = np.linspace(-10, 10, 21)
    # rl.user_actions(list(actions))

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

    episodes = 1
    num_sim = 2000
    rlist = []

    for episode in range(episodes):

        # Environment parameters
        states = np.zeros(num_sim + 1)
        states[0] = 10

        error = np.zeros(num_sim + 1)

        actions = np.zeros(num_sim + 1)
        actions[0] = 30

        # Reset the model after each episode
        tot_reward = 0

        # State and action indices
        state = 0
        action = 0

        for t in range(1, num_sim + 1):

            """
            Set-point
            """

            # if t % 1000 == 0:
            #     cur_setpoint = 20
            # else:
            #     cur_setpoint = 15
            #
            # cur_setpoint = 10
            #
            # error[t] = states[t - 1] - cur_setpoint

            """
            Disturbance
            """

            # if t == 10:
            #     model.disturbance()

            """
            RL Evaluate
            """

            if t % rl.eval_period == 0:
                state, action = rl.ucb_action_selection(states[t - 1])
                actions[t], action = rl.action_selection(state, action, 0, 25, ep_greedy=False, time=t,
                                                         min_eps_rate=1.0)
            else:
                actions[t] = actions[t - 1]

            states[t] = model(actions[t]) + np.random.uniform(-0.2, 0.2)

            """
            Feedback evaluation
            """

            if t == rl.eval_feedback:
                reward = reward_calc(states[t], 15)
                rl.matrix_update(action, reward, state, states[t], 5)
                tot_reward = tot_reward + reward

        rlist.append(tot_reward)

        rl.autosave(episode, 250)

        if episode % 100 == 0:
            print('The current error is: {:2f}'.format(np.sum(np.square(states - 15))))

        # Plotting
        plt.plot(states)
        plt.show()

    return model, rl, rlist, states, actions


# if __name__ == "__main__":
Model, RL, rList, States, Actions = simulation()
