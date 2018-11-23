import numpy as np
import matplotlib.pyplot as plt

# Load Data
x_MIMO_major = np.loadtxt('x_MIMO_major.txt')
u_MIMO_major = np.loadtxt('u_MIMO_major.txt')

x_MIMO_minor = np.loadtxt('x_MIMO_minor.txt')
u_MIMO_minor = np.loadtxt('u_MIMO_minor.txt')

x_MIMO_none = np.loadtxt('x_MIMO_none.txt')
u_MIMO_none = np.loadtxt('u_MIMO_none.txt')

x_MIMO_MPC = np.loadtxt('x_MIMO_MPC.txt')
u_MIMO_MPC = np.loadtxt('u_MIMO_MPC.txt')

x_MIMO_DDPG = np.loadtxt('x_MIMO_DDPG2_MPC.txt')
u_MIMO_DDPG = np.loadtxt('u_MIMO_DDPG2_MPC.txt')

x_MIMO_DDPG_RL = np.loadtxt('x_MIMO_DDPG_RL.txt')
u_MIMO_DDPG_RL = np.loadtxt('u_MIMO_DDPG_RL.txt')

"""
State Trajectories
"""

plt.rcParams["figure.figsize"] = (8, 5)

plt.subplot(2, 2, 1)
plt.title("State Trajectories")
# plt.plot(x_MIMO_major[:, 0], label='RL (Major Disturbances)')
# plt.plot(x_MIMO_minor[:, 0], label='RL (Minor Disturbances)')
plt.plot(x_MIMO_none[:, 0], label='RL')
plt.plot(x_MIMO_MPC[:, 0], label='MPC')
plt.plot(x_MIMO_DDPG[:, 0], label='Deep RL (MPC Cost)')
plt.plot(x_MIMO_DDPG_RL[:, 0], label='Deep RL (RL Cost)')

plt.xlim([0, 30])
plt.ylim([1, 6])

plt.legend(loc=1, prop={'size': 6})

plt.subplot(2, 2, 3)
plt.xlabel("Time, T (steps)")
# plt.plot(x_MIMO_major[:, 1], label='RL (Major Disturbances)')
# plt.plot(x_MIMO_minor[:, 1], label='RL (Minor Disturbances)')
plt.plot(x_MIMO_none[:, 1], label='RL')
plt.plot(x_MIMO_MPC[:, 1], label='MPC')
plt.plot(x_MIMO_DDPG[:, 1], label='Deep RL (MPC Cost)')
plt.plot(x_MIMO_DDPG_RL[:, 1], label='Deep RL (RL Cost)')

plt.xlim([0, 30])
plt.ylim([2, 7])

plt.legend(loc=1, prop={'size': 6})

"""
Input Trajectories
"""

plt.subplot(2, 2, 2)
plt.title("Input Trajectories")
# plt.plot(u_MIMO_major[:, 0], label='RL (Major Disturbances)')
# plt.plot(u_MIMO_minor[:, 0], label='RL (Minor Disturbances)')
plt.plot(u_MIMO_none[:, 0], label='RL')
plt.plot(u_MIMO_MPC[:, 0], label='MPC')
plt.plot(u_MIMO_DDPG[:, 0], label='Deep RL (MPC Cost)')
plt.plot(u_MIMO_DDPG_RL[:, 0], label='Deep RL (RL Cost)')

plt.xlim([0, 30])

plt.legend(loc=1, prop={'size': 6})

plt.subplot(2, 2, 4)
plt.xlabel("Time, T (steps)")
# plt.plot(u_MIMO_major[:, 1], label='RL (Major Disturbances)')
# plt.plot(u_MIMO_minor[:, 1], label='RL (Minor Disturbances)')
plt.plot(u_MIMO_none[:, 1], label='RL (No Disturbances)')
plt.plot(u_MIMO_MPC[:, 1], label='MPC')
plt.plot(u_MIMO_DDPG[:, 1], label='Deep RL (MPC Cost)')
plt.plot(u_MIMO_DDPG_RL[:, 1], label='Deep RL (RL Cost)')

plt.xlim([0, 30])

plt.legend(loc=1, prop={'size': 6})

plt.show()












