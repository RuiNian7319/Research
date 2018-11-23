import numpy as np
import sys

sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/Models')
sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/Modules')

from RL_Module import ReinforceLearning
from Linear_System_NormR import LinearSystem

"""
Define reward function for RL.  User defines the reward function structure.  The below is an example.
"""


def simulation():

    # Model Initiation
    model = LinearSystem(nsim=100, model_type='SISO', x0=np.array([0.5]), u0=np.array([1]), xs=np.array([5]),
                         us=np.array([10]), step_size=0.2)

    # model = LinearSystem(nsim=100, model_type='MIMO', x0=np.array())

    # Reinforcement Learning Initiation
    rl = ReinforceLearning(discount_factor=0.95, states_start=300, states_stop=340, states_interval=0.5,
                           actions_start=-15, actions_stop=15, actions_interval=2.5, learning_rate=0.5,
                           epsilon=0.2, doe=1.2, eval_period=1)

    """
    Example of user defined states and actions.  Users do not need to do this.  This is only if users want to define 
    their own states and actions.  RL will automatically populate states and actions if user does not input their own.
    """

    states = np.zeros([45])
    states[0:15] = np.linspace(0, 2.5, 15)
    states[15:45] = np.linspace(3, 8, 30)

    rl.user_states(list(states))

    # actions = np.zeros([20])
    # actions[0:5] = np.linspace(290, 298, 5)
    actions = np.linspace(5, 15, 41)
    # actions[30:35] = np.linspace(302, 310, 5)

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

            # if t % 30 == 0:
            #     model.x[t - 1] += - 1  # np.random.uniform(-2.3, 2.3)

            """
            RL Evaluate
            """

            if t % rl.eval_period == 0:
                state, action = rl.ucb_action_selection(model.x[t - 1, 0])
                action, action_index = rl.action_selection(state, action, model.u[t - 1, 0], no_decay=25,
                                                           ep_greedy=False, time=t,
                                                           min_eps_rate=0.5)
                # Use interpolation to perform action
                # action = rl.interpolation(model.x[t - 1, 0])
            else:
                action = model.u[t - 1, :][0]

            next_state, reward, done, info = model.step([action], t, obj_function="MPC")

            """
            Feedback evaluation
            """

            if t == rl.eval_feedback:
                rl.matrix_update(action_index, reward, state, model.x[t, 0], 5)
                tot_reward = tot_reward + reward

        rlist.append(tot_reward)

        rl.autosave(episode, 250)

        if episode % 100 == 0:
            print(model.cost_function(transient_period=120))

    return model, rl, rlist


if __name__ == "__main__":

    env, RL, rList = simulation()
