import numpy as np

class MultiHeadAttention:
    """
    class for implementing multi head causal (and masked) attention heads
    """
    def __init__(self, d_in, d_out, embedding_size, context_window, seed):

        self.num_heads = d_out // context_window

        rng = np.random.default_rng(seed=seed)

        self.wq = rng.normal(0, 1, size=(d_out, self.num_heads))
        self.wk = rng.normal(0, 1, size=(d_out, self.num_heads))
        self.wv = rng.normal(0, 1, size=(d_out, self.num_heads))

    def forward(self, batch):

        self.q = batch @ self.wq.T

if __name__ == "__main__":

    rng = np.random.default_rng(12)

    batch = rng.normal(0, 1, size = (8, 128, 256))

    att = MultiHeadAttention(5, 5, 256, 8, 0)
    