import numpy as np
import sys

sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/Models')

from RL_Module import ReinforceLearning
from Linear_System import LinearSystem

"""
Define reward function for RL.  User defines the reward function structure.  The below is an example.
"""


def simulation():

    # Model Initiation
    model = LinearSystem(nsim=100, model_type='MIMO', x0=np.array([1.333, 4]), u0=np.array([3, 6]),
                         xs=np.array([3.555, 4.666]), us=np.array([5, 7]), step_size=0.2)

    # Reinforcement Learning Initiation
    rl = ReinforceLearning(discount_factor=0.95, states_start=300, states_stop=340, states_interval=0.5,
                           actions_start=-15, actions_stop=15, actions_interval=2.5, learning_rate=0.5,
                           epsilon=0.2, doe=1.2, eval_period=1)

    """
    Example of user defined states and actions.  Users do not need to do this.  This is only if users want to define 
    their own states and actions.  RL will automatically populate states and actions if user does not input their own.
    """

    states = []

    rl.x1 = np.linspace(0, 6, 13)
    rl.x2 = np.linspace(2, 6, 9)

    for i in rl.x1:
        for j in rl.x2:
            states.append([i, j])

    rl.user_states(list(states))

    actions = []

    rl.u1 = np.linspace(1, 7, 7)
    rl.u2 = np.linspace(4, 9, 6)

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

    rlist = []

    for episode in range(1):

        # Reset the model after each episode
        model.reset(random_init=False)
        tot_reward = 0
        state = 0
        action_index = 0

        for t in range(1, model.Nsim):

            """
            Disturbance
            """

            # if t % 35 == 0:
            #     model.x[t - 1, :] += np.random.uniform(-2.1, -1.8, size=2)

            """
            RL Evaluate
            """

            if t % rl.eval_period == 0:
                state, action = rl.ucb_action_selection(model.x[t - 1, :])
                action, action_index = rl.action_selection(state, action, model.u[t - 1, :], no_decay=25,
                                                           ep_greedy=False, time=t,
                                                           min_eps_rate=0.5)
            else:
                action = model.u[t - 1, :][0]

            next_state, reward, done, info = model.step([action], t, obj_function="MPC")

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
