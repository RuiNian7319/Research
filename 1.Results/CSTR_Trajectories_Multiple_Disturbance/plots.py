import numpy as np
import matplotlib.pyplot as plt

fonts = {"family": "serif",
         "weight": "normal",
         "size": "12"}

plt.rc('font', **fonts)
plt.rc('text', usetex=True)

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

plt.rcParams["figure.figsize"] = (16, 10)

plt.subplot(2, 2, 1)
plt.ylabel("Concentration of Reactant A")
plt.title("State Trajectories")
plt.plot(x_mpc[:, 0], label='MPC')
plt.plot(x_rl_mpccost_1_action8[:, 0], label='Deep RL (MPC Cost)')
plt.plot(x_rl_rlcost[:, 0], label='Deep RL (RL Cost)')
plt.legend(frameon=False)

plt.subplot(2, 2, 3)
plt.ylabel(r"Reactor Temperature, \textit{T} (°C)")
plt.xlabel(r"Time, \textit{t} (min)")
plt.plot(x_mpc[:, 1], label='MPC')
plt.plot(x_rl_mpccost_1_action8[:, 1], label='Deep RL (MPC Cost)')
plt.plot(x_rl_rlcost[:, 1], label='Deep RL (RL Cost)')
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
plt.title(r"Input Trajectories")
plt.ylabel(r"Coolant Temperature, \textit{T} (°C)")
plt.plot(u_mpc[:, 0], label='MPC')
plt.plot(u_rl_mpccost_1_action8[:, 0], label='Deep RL (MPC Cost)')
plt.plot(u_rl_rlcost[:, 0], label='Deep RL (RL Cost)')
plt.legend(frameon=False)

plt.subplot(2, 2, 4)
plt.ylabel(r"Reactant Height, \textit{m} (m)")
plt.xlabel(r"Time, \textit{t} (min)")
plt.plot(u_mpc[:, 1], label='MPC')
plt.plot(u_rl_mpccost_1_action8[:, 1], label='Deep RL (MPC Cost)')
plt.plot(u_rl_rlcost[:, 1], label='Deep RL (RL Cost)')
plt.legend(frameon=False)

plt.savefig('States_and_Inputs_CSTR.pdf', format='pdf', dpi=1000)

plt.show()












