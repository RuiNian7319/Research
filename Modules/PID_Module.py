"""
Proportional Integral Derivative Controller in Discrete Form

November 8th, 2018
"""


def pid(set_point, x_cur, x_1, x_2, kp, ki, kd, u_1, error, ts):

    """
    Inputs
    ----
        set_point: Set-point of the controller
        x_cur: Current state
        x_1: State(t - 1)
        x_2: State(t - 2)
        kp: Proportional Gain
        ki: Integral Gain
        kd: Derivative Gain
        u_1: Last input
        error: Past error
        ts: Sampling time
    """

    ek = set_point - x_cur       # Current error
    ek_1 = set_point - x_1   # Error at time - 1
    ek_2 = set_point - x_2
    error.append(ek)        # Add to the previous error

    # ef = ek / (0.1*td + 1)

    "Discrete Time PID, derivative part is not accurate.  Ef should be at (k - 1) and (k - 2),"
    "but we don't use the derivative part"

    "From: http://users.abo.fi/htoivone/courses/isy/DiscreteTimeSystems.pdf"
    "U(k) = U(k - 1) + Kp(e(k) - e(k - 1)) + Ki * e(k) + Kd(e(k) - 2e(k - 1) + e(k - 2))"

    du = (kp * (ek - ek_1) + ki * ek + kd * (ek - 2*ek_1 + ek_2)) * ts
    u_cur = u_1 + du

    return u_cur


if __name__ == "__main__":

    pass
