import numpy as np

class MultiHeadAttention:
    """
    class for implementing multi head causal (and masked) attention heads
    """
    def init(self, d_in, d_out, embedding_size, context_window, seed):

        self.num_heads = d_out // context_window

        rng = np.random.default_rng(seed=seed)

        self.wq = rng.normal(0, 1, size=(d_out, num_heads))