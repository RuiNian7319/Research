import numpy as np
import matplotlib.pyplot as plt

# Load Data
x_mpc = np.loadtxt('x_mpc.txt')
u_mpc = np.loadtxt('u_mpc.txt')

x_rl_mpccost_1_action8 = np.loadtxt('x_rl_mpccost_1_action8.txt')
u_rl_mpccost_1_action8 = np.loadtxt('u_rl_mpccost_1_action8.txt')

x_rl_rlcost = np.loadtxt('x_rl_rlcost.txt')
u_rl_rlcost = np.loadtxt('u_rl_rlcost.txt')

"""
State Trajectories
"""

plt.rcParams["figure.figsize"] = (8, 5)

plt.subplot(2, 2, 1)
plt.ylabel("Concentration of Reactant A")
plt.title("State Trajectories")
plt.plot(x_mpc[:, 0], label='MPC')
plt.plot(x_rl_mpccost_1_action8[:, 0], label='RL (MPC Cost)')
plt.plot(x_rl_rlcost[:, 0], label='RL (RL Cost)')
plt.legend()

plt.subplot(2, 2, 3)
plt.ylabel("Reactor Temperature, T (°C)")
plt.xlabel("Time, T (min)")
plt.plot(x_mpc[:, 1], label='MPC')
plt.plot(x_rl_mpccost_1_action8[:, 1], label='RL (MPC Cost)')
plt.plot(x_rl_rlcost[:, 1], label='RL (RL Cost)')
plt.legend()

# plt.subplot(313)
# plt.plot(x_rl_1_mpccost[0:10, 2], label='test')
# plt.plot(x_rl_1_rlcost[0:10, 2], label='test')
# plt.plot(x_rl_5[0:10, 2], label='test')
# plt.plot(x_mpc_1[0:10, 2], label='test')

"""
Input Trajectories
"""

plt.subplot(2, 2, 2)
plt.title("Input Trajectories")
plt.ylabel("Coolant Temperature, T (°C)")
plt.plot(u_mpc[:, 0], label='MPC')
plt.plot(u_rl_mpccost_1_action8[:, 0], label='RL (MPC Cost)')
plt.plot(u_rl_rlcost[:, 0], label='RL (RL Cost)')
plt.legend()

plt.subplot(2, 2, 4)
plt.ylabel("Reactant Height, m (m)")
plt.xlabel("Time, T (min)")
plt.plot(u_mpc[:, 1], label='MPC')
plt.plot(u_rl_mpccost_1_action8[:, 1], label='RL (MPC Cost)')
plt.plot(u_rl_rlcost[:, 1], label='RL (RL Cost)')

plt.legend()

plt.show()












