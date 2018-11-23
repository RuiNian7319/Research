"""
Python Max Heap for Priority Replay Memory

Tutorial from Joe James with minor edits from Rui Nian to improve efficiency
"""


class MaxHeap:
    def __init__(self, items=[]):
        # Ignore index 0
        self.heap = [0]
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
    def pop(self):
        if len(self.heap) > 2:
            self.__swap(1, len(self.heap) - 1)
            maximum = self.heap.pop()
            self.__bubbleDown(1)
        elif len(self.heap) == 2:
            maximum = self.heap.pop()
        else:
            maximum = False
        return maximum

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
    m = MaxHeap([95, 3, 21, 6, 22, 62, 19, 52])
    m.push(50)
    print(str(m.heap[0:len(m.heap)]))
    print(str(m.pop()))
