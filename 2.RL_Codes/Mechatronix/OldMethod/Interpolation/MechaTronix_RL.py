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


# Model for MechaTronix
def model(action):
    """
    Action: Total pump frequency (not delta pump frequency)
    """
    return 0.01158682 * np.square(action) + 0.02409036 * action - 2.073161


# Reward function
def reward_calc(state, setpoint):
    return max(-np.square(state - setpoint), -150)


def simulation():

    # Reinforcement Learning Initiation
    rl = ReinforceLearning(discount_factor=0.95, states_start=300, states_stop=340, states_interval=0.5,
                           actions_start=-15, actions_stop=15, actions_interval=2.5, learning_rate=0.5,
                           epsilon=0.2, doe=1.2, eval_period=50)

    """
    Example of user defined states and actions.  Users do not need to do this.  This is only if users want to define 
    their own states and actions.  RL will automatically populate states and actions if user does not input their own.
    """

    states = np.linspace(-20, 20, 21)
    rl.user_states(list(states))

    actions = np.linspace(-10, 10, 21)
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

    episodes = 1
    num_sim = 2000
    rlist = []

    for episode in range(episodes):

        # Environment parameters
        env_states = np.zeros(num_sim + 1)
        env_states[0] = 40

        error = np.zeros(num_sim + 1)

        env_actions = np.zeros(num_sim + 1)
        env_actions[0] = 60

        # Reset the model after each episode
        tot_reward = 0

        # State and action indices
        state = 0
        action = 0

        if episode % 22 == 0:
            cur_setpoint = 35
        else:
            cur_setpoint = np.random.uniform(30, 60)

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
            RL Evaluate
            """

            if t % rl.eval_period == 0:
                # State: index of state; actions: Physical action; action: index of action
                state, env_actions[t], action = rl.action_selection(error[t], env_actions[t - 1],
                                                                    25, ep_greedy=False, time=t,
                                                                    min_eps_rate=0.9)

            else:
                env_actions[t] = env_actions[t - 1]

            env_states[t] = model(env_actions[t]) + np.random.uniform(-0.1, 0.1)

            """
            Feedback evaluation
            """

            if t == rl.eval_feedback:
                reward = reward_calc(env_states[t], cur_setpoint)
                rl.matrix_update(action, reward, state, error[t], 5)
                tot_reward = tot_reward + reward

        rlist.append(tot_reward)

        rl.autosave(episode, 250)

        if episode % 100 == 0:
            print('The current error is: {:2f}'.format(np.sum(np.square(env_states - cur_setpoint))))

    # Plotting
    plt.plot(env_states)
    plt.xlabel('Pump RPM, (Hertz)')
    plt.ylabel('Pressure, P')

    plt.axhline(y=cur_setpoint, color='red')

    plt.text(x=750, y=34.5, s='Set point')
    plt.text(x=750, y=35.9, s='Pressure Trajectory')

    plt.ylim([33, 43])

    plt.show()

    return model, rl, rlist, env_states, env_actions


# if __name__ == "__main__":
Model, RL, rList, States, Actions = simulation()
