import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/Models')
sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/Modules')

from RL_Module import ReinforceLearning
from SIMO_NomR import SIMOSystem

"""
Define reward function for RL.  User defines the reward function structure.  The below is an example.
"""


def simulation():

    # Model Initiation
    model = SIMOSystem(nsim=50, x0=np.array([2, 2/3]), u0=np.array([2]), xs=np.array([4, 4/3]), us=np.array([4]),
                       step_size=0.2, control=False, q_cost=1, r_cost=0.5, s_cost=0.3, random_seed=1)

    # Reinforcement Learning Initiation
    rl = ReinforceLearning(discount_factor=0.9, states_start=300, states_stop=340, states_interval=0.5,
                           actions_start=-15, actions_stop=15, actions_interval=2.5, learning_rate=0.1,
                           epsilon=0.9, doe=0, eval_period=1)

    """
    Example of user defined states and actions.  Users do not need to do this.  This is only if users want to define 
    their own states and actions.  RL will automatically populate states and actions if user does not input their own.
    """

    states = []

    x1 = np.zeros(15)
    x1[0:2] = np.linspace(2, 3, 2)
    x1[2:13] = np.linspace(3.5, 4.5, 11)
    x1[13:15] = np.linspace(5, 5.5, 2)

    x2 = np.zeros(15)
    x2[0:2] = np.linspace(0, 2 / 3, 2)
    x2[2:13] = np.linspace(1, 2, 11)
    x2[5] = 1.33
    x2[13:15] = np.linspace(2.25, 2.5, 2)

    rl.x1 = x1
    rl.x2 = x2

    for i in rl.x1:
        for j in rl.x2:
            states.append([i, j])

    rl.user_states(list(states))

    actions = np.linspace(2, 6, 17)

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
        action_index = 0

        for t in range(1, model.Nsim + 1):

            """
            Disturbance
            """

            # if t % 5 == 0:
            #     model.x[t - 1, :] += np.random.uniform(-2.1, 2.1, size=1)

            """
            RL Evaluate
            """

            if t % rl.eval_period == 0:
                state, action = rl.ucb_action_selection(model.x[t - 1, :])
                action, action_index = rl.action_selection(state, action, model.u[t - 1, :], no_decay=25,
                                                           ep_greedy=False, time=t,
                                                           min_eps_rate=0.6)

            else:
                action = model.u[t - 1, :][0]

            next_state, reward, done, info = model.step([action], t, obj_function="MPC", delta_u='l1')

            """
            Feedback evaluation
            """

            if t == rl.eval_feedback:
                rl.matrix_update(action_index, reward, state, model.x[t, :], 5)
                tot_reward = tot_reward + reward

        rlist.append(tot_reward)

        rl.autosave(episode, 250)

        if episode % 100 == 0:
            print(model.cost_function(transient_period=200))

    return model, rl, rlist


if __name__ == "__main__":

    env, RL, rList = simulation()
