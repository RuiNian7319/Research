import numpy as np


def binary_search(a_list, item):
    first = 0
    middle = 0
    last = len(a_list) - 1
    found = False

    while first <= last and not found:

        middle = (first + last) // 2

        if a_list[middle] == item:
            found = True

        else:
            if item < a_list[middle]:
                last = middle - 1

            elif item > a_list[middle]:
                first = middle + 1

            else:
                raise ValueError("Something is not right")

    return found, middle


if __name__ == "__main__":
    test_list = [4, 6, 8, 9, 11, 17, 23, 31, 55, 61, 77]
    find, item_index = binary_search(test_list, 11)

    if find:
        print(test_list[item_index])


