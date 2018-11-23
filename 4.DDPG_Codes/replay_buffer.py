""" 
Data structure for implementing experience replay
"""
from collections import deque
import random
import numpy as np


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):

        """
        The right side of the deque contains the most recent experiences

        Buffer_size: Amount of experiences that can be stored within the replay memory
        Count: Total experiences currently in replay memory
        Buffer = Data structure for storing the replay memories.
        """

        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):

        """
        s: state
        a: action
        r: reward
        t: Is the episode over or not?  Basically done.
        s2: new state
        """

        experience = (s, a, r, t, s2)

        # During replay memory population
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        # If the replay memory is full, remove the oldest memories first.
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        # Returns the amount of replay experiences in the buffer
        return self.count

    def sample_batch(self, batch_size):

        # If there is not enough memories to form one batch, take all the memories
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
    replay_buffer.add([1, 2, 3], [4, 5], 10, False, [4, 5, 6])
    replay_buffer.add([3, 7, 2], [7, 2], 5, False, [5, 5, 6])
    replay_buffer.add([6, 2, 6], [8, 5], 2, False, [4, 5, 2])
    replay_buffer.add([8, 4, 2], [2, 6], 12, False, [4, 4, 1])
    replay_buffer.add([2, 3, 8], [41, 12], 27, False, [3, 9, 5])
    replay_buffer.add([3, 1, 4], [42, 4], 31, True, [1, 5, 2])

    # Samples 5 random samples from the replay buffer
    s, a, r, t, s2 = replay_buffer.sample_batch(5)

    # Returns size of replay buffer
    replay_buffer.size()
