import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def converter(converted_data, q_matrix=np.array([])):
    k = 0

    for i in range(q_matrix.shape[0]):
        for j in range(q_matrix.shape[1]):
            converted_data[k, :] = [i, j, q_matrix[i, j]]
            k += 1

    return converted_data


if __name__ == "__main__":

    q_matrix = np.random.rand(3, 3)
    converted_data = np.zeros((q_matrix.shape[0] * q_matrix.shape[1], 3))

    convert = converter(converted_data, q_matrix)

    # Tests
    assert (q_matrix[0, 0] == convert[0, 2])
    assert (q_matrix[q_matrix.shape[0] - 1, q_matrix.shape[1] - 1] == convert[-1, 2])

    x = convert[:, 0]
    y = convert[:, 1]
    X, Y = np.meshgrid(x, y)

    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            Z[i, j] = X[i, j], Y[i, j]


    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=plt.get_cmap('hot'))
    plt.show()

