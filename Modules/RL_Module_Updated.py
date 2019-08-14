"""
Off-Policy model-free Q-learning with Upper Confidence Bound.

Rui Nian
Patch 1.04

Patch: Added Linear Interpolation
"""


import numpy as np
import random


from copy import deepcopy


class ReinforceLearning:

    """
    states_start: Min value of states
    states_end: Max value of states
    states_interval: Distance between consecutive state values
    actions_start: Min value of actions
    actions_end: Max value of actions
    actions_interval: Distance between consecutive state values
    learning_rate: The speed the Q-table is uploaded.  DEFAULT VALUE: 0.7
    epsilon: Percentage of time random action taken.    DEFAULT VALUE = 0.5
    doe: Degree of exploration for UCB.  Higher doe equates to higher exploration.  DEFAULT VALUE = 1.2
    Discount factor: Discounts future Q values due to uncertainty.   DEFAULT VALUE = 0.95
    eval_period: How many time steps between evaluation of RL.    DEFAULT VALUE = 1
    """

    def __init__(self, states_start, states_stop, states_interval, actions_start, actions_stop,
                 actions_interval, learning_rate=0.7, epsilon=0.5, doe=1.2, discount_factor=0.95, eval_period=1,
                 random_seed=None):

        self.states = list(np.arange(states_start, states_stop, states_interval * 0.99))
        self.actions = list(np.arange(actions_start, actions_stop, actions_interval * 0.99))
        self.learning_rate_0 = learning_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_0 = epsilon
        self.epsilon = epsilon
        self.doe = doe
        self.Q = np.zeros((len(self.states), len(self.actions)))
        self.NT = np.zeros((len(self.states), len(self.actions)))
        self.T = np.ones((len(self.states), len(self.actions)))
        self.eval_period = eval_period
        self.eval_feedback = 999

        # State and action lists for multiple input multiple output systems
        self.x1 = []
        self.x2 = []

        self.u1 = []
        self.u2 = []

        # Seed the results for reproducability
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    """
    Random argmax
    
    vector: input of numbers
    
    return: Index of largest number, breaking ties randomly
    """

    @staticmethod
    def rargmax(vector):

        m = np.amax(vector)
        indices = np.nonzero(vector == m)[0]

        return random.choice(indices)

    """
    Load in user defined states rather than use auto-generated states
    
    state:  User defined list of states
    """

    def user_states(self, state):
        self.states = state
        self.Q = np.zeros((len(self.states), len(self.actions)))
        self.NT = np.zeros((len(self.states), len(self.actions)))
        self.T = np.ones((len(self.states), len(self.actions)))

    """
    Load in user defined actions rather than use auto-generated actions
    
    action: User defined list of actions
    """

    def user_actions(self, action):
        self.actions = action
        self.Q = np.zeros((len(self.states), len(self.actions)))
        self.NT = np.zeros((len(self.states), len(self.actions)))
        self.T = np.ones((len(self.states), len(self.actions)))

    """
    Load in pre-trained Q, T, and NT matrices

    action: User defined list of actions
    """

    def user_matrices(self, q, t, nt):

        self.Q = q
        self.T = t
        self.NT = nt

        # Ensure the matrices have proper dimensions so RL can run
        assert(self.Q.shape == (len(self.states), len(self.actions)))
        assert(self.T.shape == (len(self.states), len(self.actions)))
        assert(self.NT.shape == (len(self.states), len(self.actions)))

    """
    Detect current state
    
    Cur_state: The current state
    
    State: The state that the current state is closest to
    """

    def state_detection(self, cur_state):

        if type(cur_state) == np.float64:

            state = min(self.states, key=lambda x_current: abs(x_current - cur_state))
            state = self.states.index(state)

        else:

            state1 = min(self.x1, key=lambda x: abs(x - cur_state[0]))
            state2 = min(self.x2, key=lambda x: abs(x - cur_state[1]))

            state = self.states.index([state1, state2])

        return state

    """
    Calculating the learning rate for Reinforcement Learning

    The decay is extremely slow and can be shown as:
    
    e = e0 / 1 + (nt^(1/10) - 1), so it takes 970,299 visits to reach a eps value of 0.01 if e0 = 1

    no_decay: Number of visits to a state-action pair before decay occurs
    sa_pair: Number of times a state action pair was visited
    min_eps_rate: The minimum epsilon rate.  Default value = 0.001
    """

    def epsilon_greedy(self, no_decay, sa_pair, min_eps_rate=0.001):

        if sa_pair < no_decay:
            pass
        else:
            self.epsilon = self.epsilon_0 / (1 + (sa_pair**(1/12) - 1))

        self.epsilon = max(self.epsilon, min_eps_rate)

    def action_selection(self, cur_state, last_input, no_decay, ep_greedy, time, min_eps_rate=0.001):

        """
        Selects an action from a list of actions.  Can be either UCB or epsilon-greedy
        state: Current state of the process
        action: Last performed action
        last_input: The last state the system was in
        no_decay:  Amount of time for learning rate and epsilon to not decay
        ep_greedy:  Whether to perform epsilon greedy action selection or not
        time: Simulation time
        min_eps_rate: The minimum epsilon rate.  Default value = 0.001
        control: New set point / value for the item being controlled
        action:  Action index preformed at current time
        """

        """
        UCB action selection portion
        """

        state = self.state_detection(cur_state)

        q_list = deepcopy(self.Q[state, :])

        for action in range(len(self.actions)):
            q_list[action] = q_list[action] + self.doe * np.sqrt(np.log(self.T[state, action]) /
                                                                 (self.NT[state, action] + 0.01))
        action = self.rargmax(q_list)

        """
        Regular action selection portion
        """

        # If epsilon greedy action is desired, calculate new epsilon value
        if ep_greedy is True:
            self.epsilon_greedy(no_decay, self.NT[state, action], min_eps_rate)
        else:
            self.epsilon = 0

        q_list = deepcopy(self.Q[state, :])

        # Returns the index of the action to be taken
        number = np.random.rand()
        if number < self.epsilon:
            action = random.randint(0, len(q_list) - 1)
        else:
            action = self.rargmax(q_list)

        # control = self.actions[action]
        control = last_input + self.actions[action]

        # Update feedback timer
        self.feedback_evaluation(time)

        return state, control, action

    """
    Calculating the learning rate for Reinforcement Learning
    
    no_decay: Number of visits to a state-action pair before decay occurs
    sa_pair: Number of times a state action pair was visited
    min_learn_rate: Minimum value for learning rate.  Default value is 0.001
    """

    def learn_rate(self, no_decay, sa_pair, min_learn_rate=0.001):

        # During no decay period
        if sa_pair < no_decay:
            pass

        # Decaying learning rate
        else:
            self.learning_rate = self.learning_rate_0 / (sa_pair - no_decay + 1)

        self.learning_rate = max(self.learning_rate, min_learn_rate)

    """
    Q-value update
    
    action: Index of the latest action
    rewards: Reward received from the latest state action pair
    old_state: The index of the state the system was in before the action was performed
    new_state: The index of the state the system is in after the action was performed
    no_decay: Amount of times state/action pair can be visited before decay in learning rate and epsilon occurs
    min_learn_rate: Minimum value for learning rate.  Default value is 0.0008
    """

    def matrix_update(self, action, rewards, old_state, cur_state, no_decay, min_learn_rate=0.0008):

        # State detection for new state
        new_state = self.state_detection(cur_state)

        # Learning rate update
        self.learn_rate(no_decay, self.NT[old_state, action], min_learn_rate)

        # Update Q matrix using the Q-learning equation
        self.Q[old_state, action] = self.Q[old_state, action] + self.learning_rate*(rewards + self.discount_factor *
                                                                                    np.max(self.Q[new_state, :])
                                                                                    - self.Q[old_state, action])
        # Update memory matrix T
        for element in range(self.T.shape[1]):
            if element != action:
                self.T[old_state, element] = self.T[old_state, element] + 1
            else:
                pass

        # Update memory matrix NT
        for j in range(self.NT.shape[1]):
            if j == action:
                self.NT[old_state, j] = self.NT[old_state, j] + 1
            else:
                pass

    """
    Determines when the next feedback evaluation period is

    time: Current simulation time
    """

    def feedback_evaluation(self, time):
        self.eval_feedback = time + self.eval_period - 1

    """
    Auto save Q, T, and NT matrices
    
    sim_time: Current time step in simulation
    time: After this many time steps, auto save the Q, T and NT matrices
    """

    def autosave(self, sim_time, time):
        if sim_time % time == 0 and sim_time != 0:
            print("Auto-saving...       Iteration number: ", sim_time)
            np.savetxt("Q_Matrix.txt", self.Q)
            np.savetxt("T_Matrix.txt", self.T)
            np.savetxt("NT_Matrix.txt", self.NT)

    """
    Linear Interpolation to get "continuous" actions
    
    y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    """

    def interpolation(self, x):
        i = 0

        # Find the state that x is less than (i.e., the upper bound of x).
        while x > self.states[i]:
            i += 1
            if i > len(self.states):
                raise ValueError("x is too big, cannot find element greater than x.")

        x0 = self.states[i - 1]
        x1 = self.states[i]

        y0_index = np.argmax(self.Q[i - 1, :])
        y1_index = np.argmax(self.Q[i, :])

        y0 = self.actions[int(y0_index)]
        y1 = self.actions[int(y1_index)]

        y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)

        return y

    """
    Weighted Linear Interpolation to use the Q-value information efficiently
    
    y = y0 + (x - x0) * (ay1 - by0) / (x1 - x0)
    
    a = eta * Q1 / Q0
    b = eta * Q0 / Q1
    """

    def weighted_interpolation(self, x, eta=1):
        i = 0
        # Find the state that x is less than (i.e., the upper bound of x).
        while x > self.states[i]:
            i += 1
            if i > len(self.states):
                raise ValueError("x is too big, cannot find element greater than x.")

        x0 = self.states[i - 1]
        x1 = self.states[i]

        y0_index = np.argmax(self.Q[i - 1, :])
        y1_index = np.argmax(self.Q[i, :])

        y0 = self.actions[int(y0_index)]
        y1 = self.actions[int(y1_index)]

        q0 = deepcopy(self.Q[i - 1, int(y0_index)])
        q1 = deepcopy(self.Q[i, int(y1_index)])

        a = eta * (q1 / (q0 + 0.001))
        b = eta * (q0 / (q1 + 0.001))

        y = y0 + (x - x0) * (a * y1 - b * y0) / (x1 - x0)

        return y

    """
    This output is used for debugging purposes.  Prints the initialization code of the RL.
    """

    def __repr__(self):
        return "ReinforceLearning(".format(len(self.states), len(self.actions))

    """
    Meaningful output if this class is printed.  Tells the users the amount of states and actions.
    """

    def __str__(self):
        return "RL controller with {} states and {} actions".format(len(self.states), len(self.actions))


class AdvantageLearning(ReinforceLearning):

    """
    Most attributes are inherited from the main reinforcement learning class.
    deltaT: Time step, this allows advantage learning to be more stable than RL in continuous processes where the change
    in Q value is very small.
    advantage: How much better is this action over other actions?  Will also be normalized where the best action has a
    0 advantage.
    """

    def __init__(self, states_start, states_stop, states_interval, actions_start, actions_stop, actions_interval,
                 deltat=1, learning_rate=0.7, epsilon=0.5, doe=0.05, discount_factor=0.95, eval_period=1):
        super().__init__(states_start, states_stop, states_interval, actions_start, actions_stop,
                         actions_interval, learning_rate, epsilon, doe, discount_factor, eval_period)
        self.deltaT = deltat
        self.advantage = np.random.uniform(-0.005, 0.005, [len(self.states), len(self.actions)])

    """
    This output is used for debugging purposes.  Prints the initialization code of the RL.
    """

    def __repr__(self):
        return "AdvantageUpdating(".format(len(self.states), len(self.actions))

    """
    Meaningful output if this class is printed.  Tells the users the amount of states and actions.
    """

    def __str__(self):
        return "Adv. Updating RL controller with {} states and {} actions".format(len(self.states), len(self.actions))

    def norm_advantage(self):

        for i in range(self.advantage.shape[0]):
            mean = np.mean(self.advantage[i, :])
            st_dev = np.std(self.advantage[i, :])

            for j in range(self.advantage.shape[1]):
                self.advantage[i, j] = (self.advantage[i, j] - mean) / st_dev

    """
    Load in user defined states rather than use auto-generated states

    state:  User defined list of states
    """

    def adv_user_states(self, state):

        super().user_states(state)
        self.advantage = np.random.uniform(-0.005, 0.005, [len(self.states), len(self.actions)])

    """
    Load in user defined actions rather than use auto-generated actions

    action: User defined list of actions
    """

    def adv_user_actions(self, action):

        super().user_actions(action)
        self.advantage = np.random.uniform(-0.005, 0.005, [len(self.states), len(self.actions)])

    """
    Load in pre-trained Q, T, and NT matrices

    action: User defined list of actions
    """

    def adv_user_matrices(self, q, t, nt, advantage):

        super().user_matrices(q, t, nt)

        self.advantage = advantage

        # Ensure the matrices have proper dimensions so RL can run
        assert (self.advantage.shape == (len(self.states), len(self.actions)))

    def adv_action_selection(self, cur_state, last_input, no_decay, ep_greedy, time, min_eps_rate=0.001):

        state = self.state_detection(cur_state)

        adv_list = deepcopy(self.advantage[state, :])

        for action in range(len(self.actions)):
            adv_list[action] = adv_list[action] + self.doe * np.sqrt(np.log(self.T[state, action]) /
                                                                     (self.NT[state, action] + 0.01))
        action = self.rargmax(adv_list)

        # If epsilon greedy action is desired, calculate new epsilon value
        if ep_greedy is True:
            self.epsilon_greedy(no_decay, self.NT[state, action], min_eps_rate)
        else:
            self.epsilon = 0

        # Returns the index of the action to be taken
        number = np.random.rand()
        if number < self.epsilon:
            action = random.randint(0, len(adv_list) - 1)
        else:
            action = action

        control = last_input + self.actions[action]

        # Update feedback timer
        self.feedback_evaluation(time)

        return control, state, action

    """
    Update the advantage matrix
    
    """

    def adv_update(self, state, action):
        self.advantage[state, action] = self.Q[state, action] - np.mean(self.Q[state, :])

    """
    If a random action is selected, the advantage matrix will be updated like this instead
    """

    def random_adv_update(self, state, action):
        self.advantage[state, action] = self.advantage[state, action] - max(self.advantage[state, :])

    """
    Update the Q, T, NT and advantage matrices
    """

    def adv_mat_update(self, action, rewards, old_state, cur_state, no_decay, min_learn_rate=0.0001):

        # State detection for new state
        new_state = self.state_detection(cur_state)

        # Learning rate update
        self.learn_rate(no_decay, self.NT[old_state, action], min_learn_rate)

        # Update Q matrix using the Q-learning equation
        self.Q[old_state, action] = self.Q[old_state, action] + self.learning_rate*(rewards + self.discount_factor *
                                                                                    np.max(self.Q[new_state, :])
                                                                                    - self.Q[old_state, action])
        self.adv_update(old_state, action)
        # self.norm_advantage(old_state)

        # Update memory matrix T
        for element in range(self.T.shape[1]):
            if element != action:
                self.T[old_state, element] = self.T[old_state, element] + 1
            else:
                pass

        # Update memory matrix NT
        for j in range(self.NT.shape[1]):
            if j == action:
                self.NT[old_state, j] = self.NT[old_state, j] + 1
            else:
                pass

    def adv_autosave(self, sim_time, time):

        super().autosave(sim_time, time)
        if sim_time % time == 0 and sim_time != 0:
            np.savetxt("Adv_Matrix.txt", self.advantage)
