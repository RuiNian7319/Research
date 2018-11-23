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
x_pid = np.loadtxt('x_pid2.txt')
u_pid = np.loadtxt('u_pid2.txt')

x_rl = np.loadtxt('x_rl.txt')
u_rl = np.loadtxt('u_rl.txt')

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

plt.plot(x_pid[0:99], label='PID')
plt.plot(x_rl[0:99], label='RL')

plt.xlim([0, 100])

plt.legend(loc=4, prop={'size': 10}, frameon=False)

"""
Input Trajectories
"""

plt.subplot(2, 1, 2)
plt.title("Input Trajectory")
plt.ylabel(r"Input, \textit{u}")
plt.xlabel(r"Time, \textit{T} (steps)")

plt.plot(u_pid[0:99], label='PID')
plt.plot(u_rl[0:99], label='RL')

plt.xlim([0, 100])

plt.legend(loc=4, prop={'size': 10}, frameon=False)

plt.show()
