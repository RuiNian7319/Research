import numpy as np
import matplotlib.pyplot as plt


def cost_function(x, u):
    cost = 0
    xs = np.array([3.555, 4.666])
    us = np.array([5, 7])
    q = 1 * np.eye(len(xs))
    r = 0.5 * np.eye(len(us))

    dx = np.subtract(x[1:-1], xs)

    du = np.subtract(u[1:-1], us)

    for i in range(dx.shape[0]):
        cost += dx[i, :] @ q @ dx[i, :].T + du[i, :] @ r @ du[i, :].T

    return cost


# Load Data
x_MIMO_MPC = np.loadtxt('x_MIMO_MPC_discounted.txt')
u_MIMO_MPC = np.loadtxt('u_MIMO_MPC_discounted.txt')

x_MIMO_DDPG_MPC = np.loadtxt('x_MIMO_DDPG_MPC5.txt')
u_MIMO_DDPG_MPC = np.loadtxt('u_MIMO_DDPG_MPC5.txt')

x_MIMO_DDPG_RL = np.loadtxt('x_MIMO_DDPG_RL.txt')
u_MIMO_DDPG_RL = np.loadtxt('u_MIMO_DDPG_RL.txt')

x_MIMO_RL = np.loadtxt('x_MIMO_none.txt')
u_MIMO_RL = np.loadtxt('u_MIMO_none.txt')

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

plt.subplot(2, 2, 1)
plt.title(r"State 1, $\textit{x}_1$")
plt.ylabel(r"Response, $\textit{x}_1$")

plt.plot(x_MIMO_MPC[:, 0], label='MPC')
plt.plot(x_MIMO_DDPG_MPC[:, 0], label='Deep RL (MPC Cost)')
# plt.plot(x_MIMO_DDPG_RL[:, 0], label='Deep RL (RL Cost)')
plt.plot(x_MIMO_RL[:, 0], label='RL (MPC Cost)')

plt.xlim([0, 30])
plt.ylim([1, 4.5])

plt.legend(prop={'size': 8}, frameon=False, loc=4)

plt.subplot(2, 2, 3)
plt.title(r"State 2, $\textit{x}_1$")
plt.ylabel(r"Response, $\textit{x}_2$")
plt.xlabel("Time, T (steps)")

plt.plot(x_MIMO_MPC[:, 1], label='MPC')
plt.plot(x_MIMO_DDPG_MPC[:, 1], label='Deep RL (MPC Cost)')
# plt.plot(x_MIMO_DDPG_RL[:, 1], label='Deep RL (RL Cost)')
plt.plot(x_MIMO_RL[:, 1], label='RL (MPC Cost)')

plt.xlim([0, 30])
plt.ylim([3, 6])

plt.legend(prop={'size': 8}, frameon=False, loc=4)

"""
Input Trajectories
"""

plt.subplot(2, 2, 2)
plt.title(r"Input 1, $\textit{u}_1$")
plt.ylabel(r"Response, $\textit{u}_1$")

plt.plot(u_MIMO_MPC[:, 0], label='MPC')
plt.plot(u_MIMO_DDPG_MPC[:, 0], label='Deep RL (MPC Cost)')
# plt.plot(u_MIMO_DDPG_RL[:, 0], label='Deep RL (RL Cost)')
plt.plot(u_MIMO_RL[:, 0], label='RL (MPC Cost)')

plt.xlim([0, 30])

plt.legend(prop={'size': 8}, frameon=False, loc=4)

plt.subplot(2, 2, 4)
plt.title(r"Input 2, $\textit{u}_2$")
plt.ylabel(r"Response, $\textit{u}_2$")
plt.xlabel("Time, T (steps)")

plt.plot(u_MIMO_MPC[:, 1], label='MPC')
plt.plot(u_MIMO_DDPG_MPC[:, 1], label='Deep RL (MPC Cost')
# plt.plot(u_MIMO_DDPG_RL[:, 1], label='Deep RL (RL Cost)')
plt.plot(u_MIMO_RL[:, 1], label='RL (MPC Cost)')

plt.xlim([0, 30])

plt.legend(prop={'size': 8}, frameon=False, loc=4)

plt.show()












