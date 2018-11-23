import numpy as np
import sys

sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/Models')
sys.path.insert(0, '/home/rui/Documents/RL_vs_MPC/Modules')

from RL_Module import ReinforceLearning
from Linear_System_SPT import LinearSystem

"""
Define reward function for RL.  User defines the reward function structure.  The below is an example.
"""


def simulation():

    # Model Initiation
    model = LinearSystem(nsim=100, model_type='SISO', x0=np.array([0.5]), u0=np.array([1]), xs=np.array([1.5]),
                         us=np.array([3]), step_size=0.2)

    # Reinforcement Learning Initiation
    rl = ReinforceLearning(discount_factor=0.90, states_start=300, states_stop=340, states_interval=0.5,
                           actions_start=-15, actions_stop=15, actions_interval=2.5, learning_rate=0.1,
                           epsilon=1, doe=0, eval_period=1)

    """
    Example of user defined states and actions.  Users do not need to do this.  This is only if users want to define 
    their own states and actions.  RL will automatically populate states and actions if user does not input their own.
    """

    # Set-point tracking errors
    states = np.zeros(39)
    states[0:5] = np.linspace(-4, -2, 5)
    states[34:39] = np.linspace(2, 4, 5)
    states[5:34] = np.linspace(-1.8, 1.8, 29)

    rl.user_states(list(states))

    actions = np.linspace(-4, 4, 33)

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

            if t % 26 == 0:
                model.xs = np.array([np.random.uniform(1, 3)])
                # model.xs = np.array([2.5])
                print(model.xs)
                pass

            """
            RL Evaluate
            """
            tracking_error = (model.x[t - 1] - model.xs)[0]

            if t % rl.eval_period == 0:
                state, action = rl.ucb_action_selection(tracking_error)
                action, action_index = rl.action_selection(state, action, model.u[t - 1, 0], no_decay=25,
                                                           ep_greedy=False, time=t,
                                                           min_eps_rate=0.25)
                # Interpolation action selection
                action = rl.interpolation(tracking_error)
            else:
                action = 0

            inputs = model.u[t - 1] + action

            next_state, reward, done, info = model.step(inputs, t, obj_function="MPC")

            """
            Feedback evaluation
            """

            if t == rl.eval_feedback:
                feedback_tracking_error = (model.x[t, 0] - model.xs)[0]
                if abs(feedback_tracking_error) > 7.5:
                    break

                rl.matrix_update(action_index, reward, state, feedback_tracking_error, 5)
                tot_reward = tot_reward + reward

        rlist.append(tot_reward)

        rl.autosave(episode, 250)

        if episode % 100 == 0:
            print(model.cost_function(transient_period=120))

    return model, rl, rlist


if __name__ == "__main__":

    env, RL, rList = simulation()
