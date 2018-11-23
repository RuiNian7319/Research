import numpy as np
import matplotlib.pyplot as plt

# Load Data
x_rl_1_mpccost = np.loadtxt('x_rl_1_mpccost.txt')
u_rl_1_mpccost = np.loadtxt('u_rl_1_mpccost.txt')

x_rl_1_rlcost = np.loadtxt('x_rl_1_rlcost.txt')
u_rl_1_rlcost = np.loadtxt('u_rl_1_rlcost.txt')

x_rl_5 = np.loadtxt('x_rl_5.txt')
u_rl_5 = np.loadtxt('u_rl_5.txt')

x_mpc_1 = np.loadtxt('x_mpc_1.txt')
u_mpc_1 = np.loadtxt('u_mpc_1.txt')

x_tabularrl_mpccost = np.loadtxt('x_tabularrl_mpccost.txt')
u_tabularrl_mpccost = np.loadtxt('u_tabularrl_mpccost.txt')

x_mpc_5 = np.loadtxt('x_mpc_5.txt')
u_mpc_5 = np.loadtxt('u_mpc_5.txt')

"""
State Trajectories
"""

plt.rcParams["figure.figsize"] = (8, 5)

plt.subplot(2, 2, 1)
plt.ylabel("Concentration of Reactant A")
plt.title("State Trajectories")
plt.plot(x_rl_1_mpccost[0:15, 0], label='RL (MPC Cost)')
plt.plot(x_rl_1_rlcost[0:15, 0], label='RL (RL Cost)')
# plt.plot(x_rl_5[0:15, 0], label='RL (Evaluate time: 5)')
plt.plot(x_mpc_1[0:15, 0], label='MPC')
# plt.plot(x_mpc_5[0:15, 0], label='MPC (Evaluation time: 5)')
plt.plot(x_tabularrl_mpccost[0:15, 0], label='Tabular RL (MPC Cost)')
plt.legend()

plt.subplot(2, 2, 3)
plt.ylabel("Reactor Temperature, T (°C)")
plt.xlabel("Time, T (min)")
plt.plot(x_rl_1_mpccost[0:15, 1], label='RL (MPC Cost)')
plt.plot(x_rl_1_rlcost[0:15, 1], label='RL (RL Cost)')
# plt.plot(x_rl_5[0:15, 1], label='RL (Evaluate time: 5)')
plt.plot(x_mpc_1[0:15, 1], label='MPC')
# plt.plot(x_mpc_5[0:15, 1], label='MPC (Evaluation time: 5)')
plt.plot(x_tabularrl_mpccost[0:15, 1], label='Tabular RL (MPC Cost)')
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
plt.plot(u_rl_1_mpccost[0:15, 0], label='RL (MPC Cost)')
plt.plot(u_rl_1_rlcost[0:15, 0], label='RL (RL Cost)')
# plt.plot(u_rl_5[0:15, 0], label='RL (Evaluate time: 5)')
plt.plot(u_mpc_1[0:15, 0], label='MPC')
# plt.plot(u_mpc_5[0:15, 0], label='MPC (Evaluation time: 5)')
plt.plot(u_tabularrl_mpccost[0:15, 0], label='Tabular RL (MPC Cost)')
plt.axis([0, 15, 299, 302.5])
plt.legend()

plt.subplot(2, 2, 4)
plt.ylabel("Reactant Height, m (m)")
plt.xlabel("Time, T (min)")
plt.plot(u_rl_1_mpccost[0:15, 1], label='RL (MPC Cost)')
plt.plot(u_rl_1_rlcost[0:15, 1], label='RL (RL Cost)')
# plt.plot(u_rl_5[0:15, 1], label='RL (Evaluate time: 5)')
plt.plot(u_mpc_1[0:15, 1], label='MPC')
# plt.plot(u_mpc_5[0:15, 1], label='MPC (Evaluation time: 5)')
plt.plot(u_tabularrl_mpccost[0:15, 1], label='Tabular RL (MPC Cost)')

plt.legend()

plt.show()












