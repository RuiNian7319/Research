import mpctools as mpc
import numpy as np
import matplotlib.pyplot as plt
from RL_Module import *
from CSTR_model import *

"""
Define reward function for RL.  User defines the reward function structure.  The below is an example.
"""


def simulation():

    def reward_calc(temp, temp_sp):
        rewards = 0

        if temp_sp * 0.999 < temp < temp_sp * 1.001:
            rewards = rewards + 15 - abs(temp - temp_sp)*20
        else:
            rewards = rewards - np.power(temp - temp_sp, 2)

        return rewards

    # Model Initiation
    model = MimoCstr(nsim=700, k0=7.2e10)

    # Reinforcement Learning Initiation
    rl = AdvantageLearning(discount_factor=0.97, states_start=300, states_stop=340, states_interval=0.5,
                           actions_start=-15, actions_stop=15, actions_interval=2.5, learning_rate=0.5,
                           epsilon=0.2, doe=1.2, eval_period=5)

    """
    Example of user defined states and actions.  Users do not need to do this.  This is only if users want to define 
    their own states and actions.  RL will automatically populate states and actions if user does not input their own.
    """

    states = np.zeros([50])
    states[0:10] = np.arange(290, 310, 20/10)
    states[10:40] = np.arange(310, 330, 20/30)
    states[40:50] = np.arange(330, 350, 20/10)

    rl.adv_user_states(list(states))

    actions = np.zeros([20])
    actions[1:20] = np.arange(-10, 10, 20 / 19)

    rl.adv_user_actions(list(actions))

    """
    Load pre-trained Q, T, NT, and advantage matrices
    """

    q = np.loadtxt("Q_Matrix.txt")
    t = np.loadtxt("T_Matrix.txt")
    nt = np.loadtxt("NT_Matrix.txt")
    adv = np.loadtxt("Adv_Matrix.txt")

    rl.adv_user_matrices(q, t, nt, adv)

    """
    Simulation portion
    """

    rlist = []

    for episode in range(701):

        # Reset the model after each episode
        model.reset(random_init=True)
        tot_reward = 0

        for t in range(1, model.Nsim + 1):

            """
            Disturbance
            """

            # if t == 10:
            #     model.disturbance()

            """
            RL Evaluate
            """

            if t % rl.eval_period == 0:
                model.u[t, 0], state, action = rl.adv_action_selection(model.x[t - 1, 1], model.u[t - 1, 0], 25,
                                                                       ep_greedy=True, time=t, min_eps_rate=0.2)
            else:
                model.u[t, :] = model.u[t - 1, :]

            model.x[t, :] = model.next_state(model.x[t - 1, :], model.u[t, :])

            """
            Feedback evaluation
            """

            if t == rl.eval_feedback:
                reward = reward_calc(model.x[t, 1], 324.5)
                rl.adv_mat_update(action, reward, state, model.x[t, 1], 5)
                tot_reward = tot_reward + reward

            if (t + 1) % 100 == 0:
                rl.norm_advantage()

        rlist.append(tot_reward)

        rl.adv_autosave(episode, 250)

        if episode % 100 == 0:
            print(model.cost_function())

    return model, rl, rlist


if __name__ == "__main__":
    Model, RL, rList = simulation()

    def hi():
        a = []
        b = []

        for i in range(RL.advantage.shape[0]):
            r = RL.advantage[i, :].tolist().index(max(RL.advantage[i, :]))
            a.append(r)

        for i in range(RL.Q.shape[0]):
            q = RL.Q[i, :].tolist().index(max(RL.Q[i, :]))
            b.append(q)

        print(np.subtract(a, b))

        return a, b
