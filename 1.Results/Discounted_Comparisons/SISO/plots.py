import numpy as np
import matplotlib.pyplot as plt


def cost_function(x, u):
    cost = 0
    xs = np.array([5])
    us = np.array([10])
    q = 1 * np.eye(len(xs))
    r = 0.5 * np.eye(len(us))

    dx = np.subtract(x[1:-1], xs)

    du = np.subtract(u[1:-1], us)

    for i in range(dx.shape[0]):
        cost += (q * dx[i] ** 2 + r * du[i] ** 2)

    return cost


# Load Data
x_SISO_MPC = np.loadtxt('x_SISO_MPC_discounted.txt')
u_SISO_MPC = np.loadtxt('u_SISO_MPC_discounted.txt')

x_SISO_DDPG_MPC = np.loadtxt('x_SISO_DDPG_MPC.txt')
u_SISO_DDPG_MPC = np.loadtxt('u_SISO_DDPG_MPC.txt')

x_SISO_DDPG_RL = np.loadtxt('x_SISO_DDPG_RL.txt')
u_SISO_DDPG_RL = np.loadtxt('u_SISO_DDPG_RL.txt')

x_SISO_RL_none = np.loadtxt('x_SISO_none.txt')
u_SISO_RL_none = np.loadtxt('u_SISO_none.txt')

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

plt.rcParams["figure.figsize"] = (8, 5)

plt.subplot(2, 1, 1)
plt.title("State Trajectory")
plt.ylabel(r"State, \textit{x}")
plt.xlabel(r"Time, \textit{T} (steps)")

plt.plot(x_SISO_MPC, label='MPC')
plt.plot(x_SISO_DDPG_MPC, label='Deep RL (MPC Cost)')
# plt.plot(x_SISO_DDPG_RL, label='Deep RL (RL Cost)')
plt.plot(x_SISO_RL_none, label='RL (MPC Cost)')

plt.ylim([0, 6])
plt.xlim([0, 8])

plt.legend(loc=4, prop={'size': 10}, frameon=False)

"""
Input Trajectories
"""

plt.subplot(2, 1, 2)
plt.title("Input Trajectory")
plt.ylabel(r"Input, \textit{u}")
plt.xlabel(r"Time, \textit{T} (steps)")

plt.plot(u_SISO_MPC, label='MPC')
plt.plot(u_SISO_DDPG_MPC, label='Deep RL (MPC Cost)')
# plt.plot(u_SISO_DDPG_RL, label='Deep RL (RL Cost)')
plt.plot(u_SISO_RL_none, label='RL (MPC Cost)')

plt.xlim([0, 8])

plt.legend(loc=4, prop={'size': 10}, frameon=False)

plt.show()
