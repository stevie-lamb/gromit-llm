"""
Files containing general utilities such
"""

import numpy as np

class Softmax:

    def __init__(self):

        pass

    def forward(self, x, axis=-1):
        """
        Each value in the list is exponentiated and then divided by the
        sum of all the other exponentiated values.

        Note: We minus the largest value for numerical stability.
        """

        x = np.array(x, dtype=np.float32)
        x_max = np.max(x, axis=axis, keepdims=True)
        x = x - x_max

        return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

    def backward(self):
        pass


class CrossEntropyLoss:

    def __init__(self):
        pass

    def forward(self, x, y):

        # assert(len(x) == len(y), "arrays must be of the same dimension")

        x, y = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

        return - 1.0 * (y @ np.log(x).T)    # y @ x.t equiv to sigma(y * log(x)) for all i

    def backward(self):
        pass



if __name__ == "__main__":

    pass