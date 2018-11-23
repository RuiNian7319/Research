import numpy as np
import matplotlib.pyplot as plt

# Load Data
x_big_none = np.loadtxt('x_big_no.txt')
x_big_inter = np.loadtxt('x_inter_big.txt')

u_big_none = np.loadtxt('u_big_no.txt')
u_big_inter = np.loadtxt('u_inter_big.txt')

x_med_none = np.loadtxt('x_medium_no.txt')
x_med_inter = np.loadtxt('x_inter_medium.txt')

u_med_none = np.loadtxt('u_medium_no.txt')
u_med_inter = np.loadtxt('u_inter_medium.txt')

x_small_none = np.loadtxt('x_small_no.txt')
x_small_inter = np.loadtxt('x_inter_small.txt')

u_small_none = np.loadtxt('u_small_no.txt')
u_small_inter = np.loadtxt('u_inter_small.txt')

x_none_no = np.loadtxt('x_none_no.txt')
x_inter_no = np.loadtxt('x_inter_no.txt')

u_none_no = np.loadtxt('u_none_no.txt')
u_inter_no = np.loadtxt('u_inter_no.txt')

"""
State Trajectories
"""

plt.rcParams["figure.figsize"] = (8, 5)

plt.subplot(2, 1, 1)
plt.title("State Trajectories")
plt.xlabel("Time, T (steps)")

plt.plot(x_big_none, label='RL')
plt.plot(x_big_inter, label='Interpolation RL')

# plt.plot(x_med_none, label='RL')
# plt.plot(x_med_inter, label='Interpolation RL')

# plt.plot(x_small_none, label='RL')
# plt.plot(x_small_inter, label='Interpolation RL')

# plt.plot(x_none_no, label='RL')
# plt.plot(x_inter_no, label='Interpolation RL')

plt.ylim([0, 5.5])
plt.xlim([0, 100])

plt.legend(loc=4, prop={'size': 6})

"""
Input Trajectories
"""

plt.subplot(2, 1, 2)
plt.title("Input Trajectories")
plt.xlabel("Time, T (steps)")

plt.plot(u_big_none, label='RL')
plt.plot(u_big_inter, label='Interpolation RL')

# plt.plot(u_med_none, label='RL')
# plt.plot(u_med_inter, label='Interpolation RL')

# plt.plot(u_small_none, label='RL')
# plt.plot(u_small_inter, label='Interpolation RL')

# plt.plot(u_none_no, label='RL')
# plt.plot(u_inter_no, label='Interpolation RL')

plt.ylim([7.5, 11.5])
plt.xlim([0, 100])

plt.legend(loc=4, prop={'size': 6})

plt.show()
