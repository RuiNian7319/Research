"""
Python Max Heap for Priority Replay Memory

Tutorial from Joe James with minor edits from Rui Nian to improve efficiency
"""


class MaxHeap:
    def __init__(self, items=[]):
        # Ignore index 0
        self.heap = [[0, 0]]
        for i in items:
            self.heap.append(i)
            self.__floatUp(len(self.heap) - 1)

    # Insert a value
    def push(self, data):
        self.heap.append(data)
        self.__floatUp(len(self.heap) - 1)

    # Get maximum value
    def peek(self):
        if self.heap[1]:
            return self.heap[1]
        else:
            return False

    # Remove maximum value
    def pop_top(self):
        if len(self.heap) > 2:
            self.__swap(1, len(self.heap) - 1)
            maximum = self.heap.pop()
            self.__bubbleDown(1)
        elif len(self.heap) == 2:
            maximum = self.heap.pop()
        else:
            maximum = False
        return maximum

    # Remove a value from the middle of the heap, given its location
    def pop_mid(self, index):
        # Swap the one you want to delete with the last element of the heap.  If the index is the last value in the list
        # don't swap because you're swapping with itself.
        if index == (len(self.heap) - 1):
            self.heap.pop()
        else:
            self.__swap(index, len(self.heap) - 1)
            self.heap.pop()

            parent = index // 2
            if self.heap[index][-1] > self.heap[parent][-1]:
                self.__floatUp(index)
            else:
                self.__bubbleDown(index)

    # Swap places of two values in the heap
    def __swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    # Ascent a recently pushed value into its proper place
    def __floatUp(self, index):
        parent = index // 2
        # If it is the only node inside the tree
        if index <= 1:
            return
        # Iteratively float up the node
        elif self.heap[index][-1] > self.heap[parent][-1]:
            self.__swap(index, parent)
            self.__floatUp(parent)

    # Used when the max value is popped off, descent a value to its proper place
    def __bubbleDown(self, index):
        # Find the two leaves
        left = index * 2
        right = index * 2 + 1
        # Assuming current value is largest
        largest = index

        # If the right leaf even exists and the current node is smaller than it
        if len(self.heap) > right and self.heap[largest][-1] < self.heap[right][-1]:
            largest = right
        # If the left leaf even exists and the current node is smaller than it
        elif len(self.heap) > left and self.heap[largest][-1] < self.heap[left][-1]:
            largest = left

        if largest != index:
            self.__swap(index, largest)
            self.__bubbleDown(largest)


if __name__ == "__main__":
    m = MaxHeap([[95, 5], [3, 6], [21, 3], [6, 7], [22, 4], [95, 15], [3, 26], [21, 33], [6, 47], [22, 50]])
    m.push([50, 44])
    print(str(m.heap))
    print(str(m.pop_top()))
