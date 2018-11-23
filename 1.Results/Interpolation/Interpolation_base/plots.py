import numpy as np
import matplotlib.pyplot as plt

# Load Data
x_big_none = np.loadtxt('x_big_none.txt')
x_big_inter = np.loadtxt('x_big_inter.txt')

u_big_none = np.loadtxt('u_big_none.txt')
u_big_inter = np.loadtxt('u_big_inter.txt')

x_med_none = np.loadtxt('x_med_none.txt')
x_med_inter = np.loadtxt('x_med_inter.txt')

u_med_none = np.loadtxt('u_med_none.txt')
u_med_inter = np.loadtxt('u_med_inter.txt')

x_small_none = np.loadtxt('x_small_none.txt')
x_small_inter = np.loadtxt('x_small_inter.txt')

u_small_none = np.loadtxt('u_small_none.txt')
u_small_inter = np.loadtxt('u_small_inter.txt')

"""
State Trajectories
"""

plt.rcParams["figure.figsize"] = (8, 5)

plt.subplot(2, 1, 1)
plt.title("State Trajectories")
plt.xlabel("Time, T (steps)")

# plt.plot(x_big_none, label='RL')
# plt.plot(x_big_inter, label='Interpolation RL')

# plt.plot(x_med_none, label='RL')
# plt.plot(x_med_inter, label='Interpolation RL')
#
plt.plot(x_small_none, label='RL')
plt.plot(x_small_inter, label='Interpolation RL')

plt.ylim([0, 5.5])
plt.xlim([0, 100])

plt.legend(loc=4, prop={'size': 6})

"""
Input Trajectories
"""

plt.subplot(2, 1, 2)
plt.title("Input Trajectories")
plt.xlabel("Time, T (steps)")

# plt.plot(u_big_none, label='RL')
# plt.plot(u_big_inter, label='Interpolation RL')

# plt.plot(u_med_none, label='RL')
# plt.plot(u_med_inter, label='Interpolation RL')

plt.plot(u_small_none, label='RL')
plt.plot(u_small_inter, label='Interpolation RL')

plt.ylim([7.5, 11.5])
plt.xlim([0, 100])

plt.legend(loc=4, prop={'size': 6})

plt.show()
