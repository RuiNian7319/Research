import mpctools as mpc
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/Models')
sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/RL_Codes')
sys.path.insert(0, '/Users/ruinian/Documents/Research/Modules')
sys.path.insert(0, '/home/rui/Documents/Research/Modules')

from RL_Module_Updated import *

"""
Define reward function for RL.  User defines the reward function structure.  The below is an example.
"""


# Model for correct system
def model_1(action1, action2):
    """
    action1: u_1 of the system
    action2: u_2 of the system
    """
    return np.square(action1) + action2 + 6


# Model for the first incorrect system
def model_2(action1, action2):
    """
    action1: u_1 of the system
    action2: u_2 of the system
    """
    return 4 * action1 + 2 * action2


# Model for the second incorrect system
def model_3(action1, action2):
    """
    action1: u_1 of the system
    action2: u_2 of the system
    """
    return 2 * action1 + 0 * action2 + 2


# Reward function
def reward_calc(state, action, setpoint):
    """
    Error based on tracking error and change in input
    """
    return max(-np.square(state - setpoint) - action, -150)


def simulation():

    # Reinforcement Learning Initiation
    rl = ReinforceLearning(discount_factor=0.95, states_start=300, states_stop=340, states_interval=0.5,
                           actions_start=-15, actions_stop=15, actions_interval=2.5, learning_rate=0.1,
                           epsilon=0.2, doe=1.2, eval_period=1)

    """
    Example of user defined states and actions.  Users do not need to do this.  This is only if users want to define 
    their own states and actions.  RL will automatically populate states and actions if user does not input their own.
    """

    # RL's state space
    states = np.linspace(-15, 15, 31)
    rl.user_states(list(states))

    # RL's action space
    actions = []

    rl.u1 = np.linspace(-5, 5, 11)
    rl.u2 = np.linspace(-5, 5, 11)

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

    episodes = 501
    num_sim = 101
    rlist = []

    for episode in range(episodes):

        # Environment parameters
        env_states = np.zeros(num_sim + 1)
        env_states[0] = 6

        error = np.zeros(num_sim + 1)

        env_actions = np.zeros((num_sim + 1, 2))
        env_actions[0] = [0, 0]

        # Reset the model after each episode
        tot_reward = 0

        # State and action indices
        state = 0
        action = 0

        if episode % 10 == 0:
            cur_setpoint = 10     # np.random.uniform(0, 60)
        else:
            cur_setpoint = 10  # np.random.uniform(2, 15)

        set_point_logs = []

        for t in range(1, num_sim + 1):

            """
            Set-point
            """

            error[t] = env_states[t - 1] - cur_setpoint

            """
            Disturbance
            """

            # if t == 10:
            #     model.disturbance()

            """
            Set-point change
            """

            if t == 30:
                cur_setpoint = 4

            if t == 60:
                cur_setpoint = 12

            if t == 85:
                cur_setpoint = 8

            set_point_logs.append(cur_setpoint)

            """
            RL Evaluate
            """

            if t % rl.eval_period == 0:
                # State: index of state; actions: Physical action; action: index of action
                state, env_actions[t], action = rl.action_selection(error[t],
                                                                    env_actions[t - 1],
                                                                    25, ep_greedy=False, time=t,
                                                                    min_eps_rate=0.1)

            else:
                env_actions[t] = env_actions[t - 1]

            # State evolution/trajectory
            env_states[t] = model_1(env_actions[t, 0], env_actions[t, 1])  # + np.random.uniform(-0.1, 0.1)

            """
            Feedback evaluation
            """

            if t == rl.eval_feedback:
                reward = reward_calc(env_states[t], np.sum(np.abs(env_actions[t - 1] - env_actions[t])), cur_setpoint)
                rl.matrix_update(action, reward, state, error[t], no_decay=10000)
                tot_reward = tot_reward + reward

        if episode % 10 == 0:
            rlist.append(tot_reward)

        rl.autosave(episode, 100)

        if episode % 100 == 0:
            print('The current error is: {:2f}'.format(np.sqrt(np.sum(np.square(env_states[0:] - cur_setpoint)))))

    # Plotting states
    fonts = {"family": "serif",
             "weight": "normal",
             "size": "12"}

    plt.rc('font', **fonts)
    plt.rc('text', usetex=True)

    plt.plot(env_states, label='State Trajectory')
    plt.plot(set_point_logs, label='Set point', color='red', linestyle='--')

    plt.xlabel(r'Time, \textit{t} (s)')
    plt.ylabel(r'Output, \textit{y}')

    plt.legend(frameon=False)

    plt.savefig('system1_traj_state.pdf', format='pdf', dpi=750)

    plt.show()

    # Plot actions
    plt.plot(env_actions[:, 0], label=r'$u_1$ Trajectory')
    plt.plot(env_actions[:, 1], label=r'$u_2$ Trajectory')

    plt.xlabel(r'Time, \textit{t} (s)')
    plt.ylabel(r'Control actions, \textit{u}')

    plt.legend(frameon=False)

    plt.savefig('system1_traj_input.pdf', format='pdf', dpi=750)

    plt.show()

    return model_2, rl, rlist, env_states, env_actions, set_point_logs


# if __name__ == "__main__":
Model, RL, rList, States, Actions, Setpoints = simulation()
