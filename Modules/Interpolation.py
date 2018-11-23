import numpy as np


def interpolation(x, a_array):
    state_list = [1, 2, 3, 4, 5]
    action_list = [5, 2, 4, 3, 7]

    i = 0
    while x > state_list[i]:
        i += 1
        if i > len(state_list):
            raise ValueError("x is too big, cannot find element greater than x.")

    x0 = state_list[i - 1]
    x1 = state_list[i]

    y0_index = np.argmax(a_array[i - 1, :])
    y1_index = np.argmax(a_array[i, :])

    y0 = action_list[int(y0_index)]
    y1 = action_list[int(y1_index)]

    y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    return y


if __name__ == "__main__":

    test_array = np.array([[1, 2, 3, 4, 5], [2, 4, 6, 4, 2], [3, 6, 2, 4, 8], [4, 5, 6, 4, 9], [5, 8, 0, 5, 4]])

    Value = interpolation(2.5, test_array)
