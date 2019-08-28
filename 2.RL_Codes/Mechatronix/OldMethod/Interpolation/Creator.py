import numpy as np


def splitter(values):
    states = np.zeros(307)

    for i in range(1, len(values)):
        states[6 * i] = values[i - 1]

    for i in range(1, len(values)):
        states[6 * i + 1:6 * (i + 1)] = np.linspace(states[6 * i], states[6 * (i + 1)], 5) + np.random.normal(0, 0.05)

    plt.plot(states[6:300])
