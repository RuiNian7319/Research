import numpy as np


def splitter(values):
    states = np.zeros(307)

    for i in range(1, len(values)):
        states[6 * i] = values[i - 1]

    for i in range(1, len(values)):
        states[6 * i + 1:6 * (i + 1)] = np.linspace(states[6 * i], states[6 * (i + 1)], 5) + np.random.normal(0, 0.05)

    output = np.array([states, np.linspace(-5, 301, 307)])

    plt.plot(output[0, 6:300])
    plt.show()

    np.savetxt('set_1.csv', output[:, 6:300].T, delimiter=',')

    return output
