import numpy as np
import matplotlib.pyplot as plt

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab


class OrnsteinUhlenbeckActionNoise:

    """
    Introduces time correlated noise into the action term taken by the deterministic policy.
    he Ornstein-Uhlenbeck process satisfies the following stochastic differential equation:

        dxt = theta*(mu - xt)*dt + sigma*dWt

    where dWt can be simplified into sqrt(dt)*N(0, 1), i.e., white noise.

    Mu: Mean to be arrived at in the end of the process
    Sigma:  Amount of noise injected into the system.  Volatility of average magnitude
    Theta: How much weight towards going towards the mean, mu.  Rate of mean reversion.
    dt: Sampling time
    """

    def __init__(self, mu, sigma=0.3, theta=.15, dt=1, random_seed=1, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

        self.decay_amount = None

        np.random.seed(random_seed)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def noise_decay(self, decay_rate):
        if self.decay_amount is None:
            self.decay_amount = self.sigma / decay_rate
        else:
            pass

        if -0.99 * self.decay_amount < self.sigma < 0.99 * self.decay_amount:
            pass
        else:
            self.sigma -= self.decay_amount

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


if __name__ == '__main__':

    actor_noise = OrnsteinUhlenbeckActionNoise(np.array([4]), dt=1)

    noise_trajectory = []
    for i in range(1000):
        noise_trajectory.append(actor_noise())
        actor_noise.noise_decay(1000)

    plt.plot(noise_trajectory)
    plt.show()
