import mpctools as mpc
import numpy as np
import matplotlib.pyplot as plt
import random

"""
A cstr model with 3 states and 2 inputs.  The controlled variables are the concentration of reactant a (c0), the height
inside the CSTR, and the temperature inside the CSTR.  The manipulated variables are the coolant temperature and the
outlet flow rate.  The inlet flow rate is identified as a random disturbance.
"""


class MimoCstr:

    """
    delta = Model discretization size for CasaDI
    Nsim = Length of simulation
    Nx = Number of states
    Nu = Number of inputs
    Nt = Prediction and Control horizon
    F0 = Initial flow rate, m3/min
    T0 = Initial temperature, K
    c0 = Initial concentration of reactant A, kmol/m3
    r = Radius of reactor, m
    k0 = Reaction coefficient, min-1
    E/R = Activation energy divided by gas constant, K
    U = kJ/min•m2•K
    rho = Density, kg/m3
    Cp = Specific heat capacity, kJ/kg•K
    dH = Energy released by reaction, kJ/kmol
    xs = Steady state states
    us = Steady state inputs
    x0 = Initial states
    """

    def __init__(self, nsim, delta=1, nx=3, nu=2, nt=10, f0=0.1, t0=350, c0=1, r=0.219, k0=7.2e10, er_ratio=8750,
                 u=54.94, rho=1000, cp=0.239, dh=-5e4, xs=np.array([0.878, 324.5, 0.659]), us=np.array([300, 0.1]),
                 x0=np.array([1, 310, 0.659]), control=False, q_cost=0.1, r_cost=0.1):

        self.Delta = delta
        self.Nsim = nsim
        self.Nx = nx
        self.Nu = nu
        self.Nt = nt
        self.F0 = f0
        self.T0 = t0
        self.c0 = c0
        self.r = r
        self.k0 = k0
        self.er_ratio = er_ratio
        self.U = u
        self.rho = rho
        self.Cp = cp
        self.dH = dh
        self.xs = xs
        self.us = us
        self.xsp = np.zeros([self.Nsim + 1, self.Nx])
        self.xsp[0, :] = xs
        self.x0 = x0
        self.x = np.zeros([self.Nsim + 1, self.Nx])
        self.x[0, :] = self.x0
        self.u = np.zeros([self.Nsim + 1, self.Nu])
        self.u[0, :] = [300, 0.1]
        self.u[:, 1] = 0.1
        self.control = control

        self.cstr_sim = mpc.DiscreteSimulator(self.ode, self.Delta, [self.Nx, self.Nu], ["x", "u"])
        self.cstr_ode = mpc.getCasadiFunc(self.ode, [self.Nx, self.Nu], ["x", "u"], funcname="odef")

        # Cost function Q and R tuning parameters
        self.q_cost = q_cost
        self.r_cost = r_cost

    """
    This output is used for debugging purposes.  Prints the initialization code of the RL.
    """

    def __repr__(self):
        return "MimoCstr({}, {}, {}, {}, {})".format(self.Delta, self.Nsim, self.Nx, self.Nu, self.Nt)

    """
    Meaningful output if this class is printed.  Tells the users a description of the object.
    """

    def __str__(self):
        return "Simulator for a 3 state, 2 input CSTR."

    """
    Ordinary differential equation of the CSTR
    
    states: The states
    inputs: The controller inputs
    """

    def ode(self, state, inputs):
        c = state[0]
        t = state[1]
        h = state[2]

        tc = inputs[0]
        f = 0.1   # inputs[1]

        rate = self.k0 * c * np.exp(-self.er_ratio / t)

        dxdt = [
            self.F0 * (self.c0 - c) / (np.pi * self.r ** 2 * h) - rate,
            self.F0 * (self.T0 - t) / (np.pi * self.r ** 2 * h) - self.dH / (self.rho * self.Cp) * rate + 2 * self.U /
            (self.r * self.rho * self.Cp) * (tc - t),
            (self.F0 - f) / (np.pi * self.r ** 2)
        ]

        if self.control is True:
            dxdt.append(np.zeros(int(self.Nx * 0.5)))

        return np.array(dxdt)

    """
    Calculates the next states of the simulation

    states: The states
    inputs: The controller inputs
    """

    def next_state(self, states, inputs):
        return self.cstr_sim.sim(states, inputs)

    """
    Resets the CSTR simulation to a clean slate
    """

    def reset(self, random_init=True):
        self.x = np.zeros((self.Nsim + 1, self.Nx))
        self.x[0, :] = self.x0

        self.u = np.zeros((self.Nsim + 1, self.Nu))

        if random_init is True:
            self.u[0, :] = [np.random.uniform(295, 345), 0.1]
        else:
            self.u[0, :] = [300, 0.1]

        self.u[:, 1] = 0.1

    """
    Controller performance assessment using integral of absolute error, integral of squared error, and MPC Error
    
    If x is True, that means user wants to calculate the cost of a user specified x
    
    x: User specified state matrix
    error: Type of error calculated
    transient_period: Period to calculate the cost for MPC cost
    """

    def cost_function(self, x=None, error="MPC", transient_period=15):
        cost = 0

        if x is None:
            if error == "IAE":
                cost = sum(abs(self.x[1:] - self.xs)) / self.Nsim
            elif error == "ISE":
                cost = np.power(self.x[1:] - self.xs, 2)
                cost = sum(cost) / self.Nsim
            elif error == "MPC":
                if self.control is True:
                    nx = int(self.Nx / 2)
                else:
                    nx = int(self.Nx)

                dx = np.subtract(self.x[1:transient_period, 0:nx], self.xs[0:nx])

                Q = self.q_cost * np.eye(nx)

                du = np.subtract(self.u[1:transient_period, 0:self.Nu], self.us)
                R = self.r_cost * np.eye(self.Nu)

                for i in range(dx.shape[0]):
                    cost += dx[i, :] @ Q @ dx[i, :].T + du[i, :] @ R @ du[i, :].T

            else:
                print("Error 1: Unspecified Loss Type")

        else:

            assert (x.shape[1] == len(self.xs))

            if error == "IAE":
                cost = sum(abs(x[1:] - self.xs)) / self.Nsim
            elif error == "ISE":
                cost = np.power(x[1:] - self.xs, 2)
                cost = sum(cost) / self.Nsim
            elif error == "MPC":
                if self.control is True:
                    nx = self.Nx / 2
                else:
                    nx = self.Nx
                dx = np.subtract(self.x[1:transient_period, 0:nx], self.xs[0:nx])

                Q = self.q_cost * np.eye(nx)

                du = np.subtract(self.u[1:transient_period, 0:self.Nu], self.us)
                R = self.r_cost * np.eye(self.Nu)

                for i in range(dx.shape[0]):
                    cost += dx[i, :] @ Q @ dx[i, :].T + du[i, :] @ R @ du[i, :].T
            else:
                print("Error 1: Unspecified Loss Type")

        return cost

    def disturbance(self):
        # 10% increase in flow rate
        self.F0 = self.F0 * 1.1

        # Rebuild the simulator
        self.cstr_sim = mpc.DiscreteSimulator(self.ode, self.Delta, [self.Nx, self.Nu], ["x", "u"])

    def plots(self):
        plt.rcParams["figure.figsize"] = (5, 8)

        plt.subplot(311)
        plt.plot(self.x[:, 0])

        plt.subplot(312)
        plt.plot(self.x[:, 1])

        plt.subplot(313)
        plt.plot(self.x[:, 2])


if __name__ == "__main__":

    cstr = MimoCstr(nsim=50, nx=6, xs=np.array([0.878, 324.5, 0.659, 0, 0, 0]),
                    x0=np.array([1, 310, 0.659, 0, 0, 0]), control=True)
    cstr.x[0, :] = cstr.xs

    for time in range(1, cstr.Nsim + 1):
        cstr.u[time, :] = cstr.us
        cstr.x[time, :] = cstr.next_state(cstr.x[time - 1, :], cstr.u[time - 1, :])

        # if time == 30:
        #     cstr.disturbance()
