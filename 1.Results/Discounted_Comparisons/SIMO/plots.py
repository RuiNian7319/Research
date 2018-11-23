import numpy as np
import matplotlib.pyplot as plt

# Load Data
x_SIMO_MPC = np.loadtxt('x_SIMO_MPC.txt')
u_SIMO_MPC = np.loadtxt('u_SIMO_MPC.txt')

x_SIMO_DDPG_MPC = np.loadtxt('x_SIMO_DDPG_MPC.txt')
u_SIMO_DDPG_MPC = np.loadtxt('u_SIMO_DDPG_MPC.txt')

x_SIMO_DDPG_RL = np.loadtxt('x_SIMO_DDPG_RL.txt')
u_SIMO_DDPG_RL = np.loadtxt('u_SIMO_DDPG_RL.txt')
#
x_SIMO_RL = np.loadtxt('x_SIMO_RL.txt')
u_SIMO_RL = np.loadtxt('u_SIMO_RL.txt')

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
plt.rcParams["figure.figsize"] = (10, 5)

plt.subplot(1, 2, 1)
plt.title(r"State 1, $\textit{x}_1$", fontsize=20)
plt.xlabel(r"Time, \textit{T} (steps)")
plt.ylabel(r"Response, $\textit{x}_1$")

plt.plot(x_SIMO_MPC[:, 0], label='MPC')
plt.plot(x_SIMO_DDPG_MPC[:, 0], label='Deep RL (MPC Cost)')
# plt.plot(x_SIMO_DDPG_RL[:, 0], label='Deep RL (RL Cost)')
plt.plot(x_SIMO_RL[:, 0], label='RL (MPC Cost)')

plt.xlim([0, 30])
plt.ylim([1, 4.5])

plt.legend(prop={'size': 10}, frameon=False, loc=4)

plt.subplot(1, 2, 2)
plt.title(r"State 2, $\textit{x}_2$", fontsize=20)
plt.xlabel(r"Time, \textit{T} (steps)")
plt.ylabel(r"Response, $\textit{x}_2$")

plt.plot(x_SIMO_MPC[:, 1], label='MPC')
plt.plot(x_SIMO_DDPG_MPC[:, 1], label='Deep RL (MPC Cost)')
# plt.plot(x_SIMO_DDPG_RL[:, 1], label='Deep RL (RL Cost)')
plt.plot(x_SIMO_RL[:, 1], label='RL (MPC Cost)')

plt.xlim([0, 30])

plt.legend(prop={'size': 10}, frameon=False, loc=4)

plt.show()

"""
Input Trajectories
"""

plt.figure()

plt.title(r"Input Trajectory, \textit{u}", fontsize=28)

plt.plot(u_SIMO_MPC, label='MPC')
plt.plot(u_SIMO_DDPG_MPC, label='Deep RL (MPC Cost)')
# plt.plot(u_SIMO_DDPG_RL, label='Deep RL (RL Cost)')
plt.plot(u_SIMO_RL, label='RL (MPC Cost)')

plt.xlim([0, 30])

plt.legend(prop={'size': 15}, frameon=False, loc=4)


plt.show()












