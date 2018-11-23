import numpy as np
import matplotlib.pyplot as plt

# Load Data
x_MISO_MPC = np.loadtxt('x_MISO_MPC.txt')
u_MISO_MPC = np.loadtxt('u_MISO_MPC.txt')

x_MISO_DDPG_MPC = np.loadtxt('x_MISO_DDPG_MPC.txt')
u_MISO_DDPG_MPC = np.loadtxt('u_MISO_DDPG_MPC.txt')
#
# x_MIMO_DDPG_RL = np.loadtxt('x_MIMO_DDPG_RL.txt')
# u_MIMO_DDPG_RL = np.loadtxt('u_MIMO_DDPG_RL.txt')
#
x_MISO_RL = np.loadtxt('x_MISO_RL.txt')
u_MISO_RL = np.loadtxt('u_MISO_RL.txt')


"""
Math Plotting Library settings
"""

fonts = {"family": "serif",
         "weight": "normal",
         "size": "12"}

plt.rc('font', **fonts)
plt.rc('text', usetex=True)

"""
State Trajectories
"""

plt.figure()
plt.title(r"State Trajectory, \textit{x}", fontsize=28)

plt.xlabel(r"Time, \textit{T} (steps)")
plt.ylabel(r"Response, \textit{y} (steps)")

plt.plot(x_MISO_MPC, label='MPC')
plt.plot(x_MISO_DDPG_MPC, label='Deep RL (MPC Cost)')
# plt.plot(x_MIMO_DDPG_RL[:, 0], label='Deep RL (RL Cost)')
plt.plot(x_MISO_RL, label='RL (MPC Cost)')

plt.xlim([0, 30])
plt.ylim([2, 5])

plt.legend(prop={'size': 15}, frameon=False, loc=4)

plt.show()

"""
Input Trajectories
"""
plt.rcParams["figure.figsize"] = (10, 5)
plt.gca().set_title('title')

plt.subplot(1, 2, 1)
plt.title(r"Input 1, $\textit{u}_1$", fontsize=28)

plt.xlabel(r"Time, \textit{T} (steps)")
plt.ylabel(r"Input, $\textit{u}_1$")

plt.plot(u_MISO_MPC[:, 0], label='MPC')
plt.plot(u_MISO_DDPG_MPC[:, 0], label='Deep RL (MPC Cost)')
# plt.plot(u_MIMO_DDPG_RL[:, 0], label='Deep RL (RL Cost)')
plt.plot(u_MISO_RL[:, 0], label='RL (MPC Cost)')

plt.xlim([0, 30])

plt.legend(prop={'size': 15}, frameon=False, loc=4)

plt.subplot(1, 2, 2)
plt.title(r"Input 2, $\textit{u}_2$", fontsize=28)
plt.xlabel(r"Time, \textit{T} (steps)")

plt.plot(u_MISO_MPC[:, 1], label='MPC')
plt.plot(u_MISO_DDPG_MPC[:, 1], label='Deep RL (MPC Cost)')
# plt.plot(u_MIMO_DDPG_RL[:, 1], label='Deep RL (RL Cost)')
plt.plot(u_MISO_RL[:, 1], label='RL (MPC Cost)')

plt.xlim([0, 30])

plt.legend(prop={'size': 15}, frameon=False, loc=4)

plt.show()












