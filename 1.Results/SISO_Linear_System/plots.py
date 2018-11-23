import numpy as np
import matplotlib.pyplot as plt

# Load Data
x_SISO_major = np.loadtxt('x_SISO_major.txt')
u_SISO_major = np.loadtxt('u_SISO_major.txt')

x_SISO_minor = np.loadtxt('x_SISO_minor.txt')
u_SISO_minor = np.loadtxt('u_SISO_minor.txt')

x_SISO_none = np.loadtxt('x_SISO_none.txt')
u_SISO_none = np.loadtxt('u_SISO_none.txt')

x_SISO_MPC = np.loadtxt('x_SISO_MPC.txt')
u_SISO_MPC = np.loadtxt('u_SISO_MPC.txt')

x_SISO_DDPG_MPC = np.loadtxt('x_SISO_DDPG_MPC.txt')
u_SISO_DDPG_MPC = np.loadtxt('u_SISO_DDPG_MPC.txt')

x_SISO_DDPG_RL = np.loadtxt('x_SISO_DDPG_RL.txt')
u_SISO_DDPG_RL = np.loadtxt('u_SISO_DDPG_RL.txt')

"""
State Trajectories
"""

plt.rcParams["figure.figsize"] = (8, 5)

plt.subplot(2, 1, 1)
plt.title("State Trajectories")
plt.ylabel("State, x")
plt.xlabel("Time, T (steps)")
# plt.plot(x_SISO_major, label='RL (Major Disturbances)')
# plt.plot(x_SISO_minor, label='RL (Minor Disturbances)')
plt.plot(x_SISO_none, label='RL')
plt.plot(x_SISO_MPC, label='MPC')
plt.plot(x_SISO_DDPG_MPC, label='Deep RL (MPC Cost)')
plt.plot(x_SISO_DDPG_RL, label='Deep RL (RL Cost)')
plt.ylim([0, 6])
plt.xlim([0, 8])

plt.legend(loc=4, prop={'size': 6})

"""
Input Trajectories
"""

plt.subplot(2, 1, 2)
plt.title("Input Trajectories")
plt.ylabel("State, x")
plt.xlabel("Time, T (steps)")
# plt.plot(u_SISO_major, label='RL (Major Disturbances)')
# plt.plot(u_SISO_minor, label='RL (Minor Disturbances)')
plt.plot(u_SISO_none, label='RL')
plt.plot(u_SISO_MPC, label='MPC')
plt.plot(u_SISO_DDPG_MPC, label='Deep RL (MPC Cost)')
plt.plot(u_SISO_DDPG_RL, label='Deep RL (RL Cost)')
plt.xlim([0, 8])

plt.legend(loc=4, prop={'size': 6})

plt.show()
