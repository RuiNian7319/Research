""" 
Data structure for implementing experience replay

Prioritized Experience Replay Referenced from: https://arxiv.org/pdf/1511.05952.pdf

Author: Rui Nian
"""
import random
import numpy as np

from collections import deque
from MaxHeap import MaxHeap


class ReplayBuffer:

    def __init__(self, buffer_size, random_seed=1, alpha=0.7, beta=0.5, epsilon=0.001):

        """
        The right side of the deque contains the most recent experiences

        Buffer_size: Amount of experiences that can be stored within the replay memory
        Count: Total experiences currently in replay memory
        Buffer: Data structure for storing the replay memories.

        alpha: Determines the degree of prioritization utilized.  Alpha = 0 refers to the uniform case.
        beta: Non-uniform probability correction factor.  Beta = 1 will fully correct the non-uniform probability.
        epsilon: Small positive integer for proportional based replay to ensure experiences with low errors have > 0%
                 probability of being visited again
        """

        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

        # Priority Replay Characteristics
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        random.seed(random_seed)

    def add(self, s, a, r, t, s2, td_error):

        """
        s: state
        a: action
        r: reward
        t: Is the episode over or not?  Basically done.
        s2: new state
        td_error: Temporal Difference Error:  TD Error = [r + gamma*argmax(a) Q(s', a')] - Q(s, a)
        """

        experience = (s, a, r, t, s2, td_error)

        # During replay memory population
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1

        # If the replay memory is full, remove the oldest memories first.
        else:
            self.buffer.popleft()
            self.buffer.append(experience)






    def priority_sample_batch(self, batch_size, type='rank'):

        """
        batch_size: The size of mini-batches used for optimization
        type: Type of priority experience replay.  Can be either proportional or rank.
        """

        # If there are not enough memories to form one batch, take all the memories
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)

        # If proportional priority replay is selected
        elif type == "proportional":
            batch = []

        # If rank priority replay is selected
        elif type == "rank":

            for sample in self.buffer[5]:
                sample = sample / sample.location()

            batch = []

        # If incorrect selection, just do normal experience replay
        else:
            print("Invalid priority replay method selected.  Performing uniform experience replay instead.")
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])
        td_batch = np.array([_[5] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch, td_batch






    def sample_batch(self, batch_size):

        # If there are not enough memories to form one batch, take all the memories
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)

        # If there are enough, sample batch number of memories
        else:
            batch = random.sample(self.buffer, batch_size)

        # Format the sampled batch into s, a, r, done, s2.
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def size(self):
        # Returns the amount of replay experiences in the buffer
        return self.count

    def clear(self):
        # Clear the replay memory
        self.buffer.clear()
        self.count = 0


if __name__ == "__main__":

    """
    Example of the replay buffer
    """

    # Initiate the replay buffer
    replay_buffer = ReplayBuffer(10)

    # Populate the replay buffer with random junk
    replay_buffer.add([1, 2, 3], [4, 5], 10, False, [4, 5, 6], -10)
    replay_buffer.add([3, 7, 2], [7, 2], 5, False, [5, 5, 6], 20)
    replay_buffer.add([6, 2, 6], [8, 5], 2, False, [4, 5, 2], -5)
    replay_buffer.add([8, 4, 2], [2, 6], 12, False, [4, 4, 1], 1)
    replay_buffer.add([2, 3, 8], [41, 12], 27, False, [3, 9, 5], -100)
    replay_buffer.add([3, 1, 4], [42, 4], 31, True, [1, 5, 2], 120)

    # Samples 5 random samples from the replay buffer
    replay_buffer.priority_sample_batch(5, 'rank')
    S, A, R, T, S2 = replay_buffer.sample_batch(5)

    # Returns size of replay buffer
    replay_buffer.size()
