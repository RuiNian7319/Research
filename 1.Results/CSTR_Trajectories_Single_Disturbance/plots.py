import numpy as np
import matplotlib.pyplot as plt

fonts = {"family": "serif",
         "weight": "normal",
         "size": "12"}

plt.rc('font', **fonts)
plt.rc('text', usetex=True)

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

plt.rcParams["figure.figsize"] = (16, 10)

plt.subplot(2, 2, 1)
plt.ylabel("Concentration of Reactant A")
plt.title("State Trajectories")
# plt.plot(x_rl_1_mpccost[0:15, 0], label='Deep RL (MPC Cost)')
# plt.plot(x_rl_1_rlcost[0:15, 0], label='Deep RL (RL Cost)')
plt.plot(x_rl_5[0:15, 0], label='Deep RL (Evaluate time: 5)')
# plt.plot(x_mpc_1[0:15, 0], label='MPC')
plt.plot(x_mpc_5[0:15, 0], label='MPC (Evaluation time: 5)')
plt.plot(x_tabularrl_mpccost[0:15, 0], label='Tabular RL (Evaluation time: 5)')
plt.legend(frameon=False)

plt.subplot(2, 2, 3)
plt.ylabel(r"Reactor Temperature, \textit{T} (°C)")
plt.xlabel(r"Time, \textit{t} (min)")
# plt.plot(x_rl_1_mpccost[0:15, 1], label='Deep RL (MPC Cost)')
# plt.plot(x_rl_1_rlcost[0:15, 1], label='Deep RL (RL Cost)')
plt.plot(x_rl_5[0:15, 1], label='Deep RL (Evaluate time: 5)')
# plt.plot(x_mpc_1[0:15, 1], label='MPC')
plt.plot(x_mpc_5[0:15, 1], label='MPC (Evaluation time: 5)')
plt.plot(x_tabularrl_mpccost[0:15, 1], label='Tabular RL (Evaluation time: 5)')
plt.legend(frameon=False)

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
plt.ylabel(r"Coolant Temperature, \textit{T} (°C)")
# plt.plot(u_rl_1_mpccost[0:15, 0], label='Deep RL (MPC Cost)')
# plt.plot(u_rl_1_rlcost[0:15, 0], label='Deep RL (RL Cost)')
plt.plot(u_rl_5[0:15, 0], label='Deep RL (Evaluate time: 5)')
# plt.plot(u_mpc_1[0:15, 0], label='MPC')
plt.plot(u_mpc_5[0:15, 0], label='MPC (Evaluation time: 5)')
plt.plot(u_tabularrl_mpccost[0:15, 0], label='Tabular RL (Evaluation time: 5)')
plt.axis([0, 15, 299, 302.5])
plt.legend(frameon=False)

plt.subplot(2, 2, 4)
plt.ylabel(r"Reactant Height, \textit{m} (m)")
plt.xlabel(r"Time, \textit{t} (min)")
# plt.plot(u_rl_1_mpccost[0:15, 1], label='Deep RL (MPC Cost)')
# plt.plot(u_rl_1_rlcost[0:15, 1], label='Deep RL (RL Cost)')
plt.plot(u_rl_5[0:15, 1], label='Deep RL (Evaluate time: 5)')
# plt.plot(u_mpc_1[0:15, 1], label='MPC')
plt.plot(u_mpc_5[0:15, 1], label='MPC (Evaluation time: 5)')
plt.plot(u_tabularrl_mpccost[0:15, 1], label='Tabular RL (Evaluation time: 5)')

plt.legend(frameon=False)

# plt.savefig('States_and_Inputs_1_CSTR.pdf', format='pdf', dpi=1000)
plt.savefig('States_and_Inputs_5_CSTR.pdf', format='pdf', dpi=1000)

plt.show()












