import numpy as np


class OrnsteinUhlenbeckActionNoise:

    """
    Introduces time correlated noise into the action term taken by the deterministic policy.
    he Ornstein-Uhlenbeck process satisfies the following stochastic differential equation:

        dxt = theta*(mu - xt)*dt + sigma*dWt

    where dWt can be simplified into dt*N(0, 1), i.e., white noise.

    Mu: Mean to be arrived at in the end of the process
    Sigma:  Amount of noise injected into the system.  Volatility of average sigma mean.
    Theta: How much weight towards going towards the mean, mu.  Rate of mean reversion.
    dt: Sampling time
    """

    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = np.array(sigma)
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __call__(self):

        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


if __name__ == "__main__":
    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(2), sigma=[1, 0.0008], theta=[0.15, 0.15], dt=[1e-2, 1e-2])
