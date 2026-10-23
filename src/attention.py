import numpy as np
from utilities import Softmax

class MultiHeadAttention:
    """
    class for implementing multi head causal (and masked) attention heads
    """
    def __init__(self, dropout, num_heads, embedding_size, context_window, seed):


        self.dropout = dropout
        self.d_in = embedding_size
        self.d_out = embedding_size
        self.embedding_size = embedding_size
        self.context_window = context_window

        self.sm = Softmax()

        assert (self.d_out % num_heads == 0, "d_out must be divisible by num_heads")


        self.num_heads = num_heads
        self.head_dim = self.embedding_size // self.num_heads

        rng = np.random.default_rng(seed=seed)

        self.wq = rng.normal(0, 1, size=(self.d_in, self.d_out))
        self.wk = rng.normal(0, 1, size=(self.d_in, self.d_out))
        self.wv = rng.normal(0, 1, size=(self.d_in, self.d_out))

        self.wo = rng.normal(0, 1, size=(self.d_out, self.d_out))

    def forward(self, batch):

        batches, _, _ = batch.shape

        self.q = batch @ self.wq.T
        self.k = batch @ self.wk.T
        self.v = batch @ self.wv.T

        #reshape and transpose
        self.q = self.q.reshape(batches, self.context_window, self.num_heads, self.head_dim)
        self.k = self.k.reshape(batches, self.context_window, self.num_heads, self.head_dim)
        self.v = self.v.reshape(batches, self.context_window, self.num_heads, self.head_dim)

        self.q = self.q.transpose((0, 2, 1, 3))
        self.k = self.k.transpose((0, 2, 3, 1))
        self.v = self.v.transpose((0, 2, 1, 3))

        self.att = self.q @ self.k

        self.att = np.tril(self.att)

        # prepare for softmax
        self.att[self.att == 0.0] = -np.inf

        # apply softmax
        self.att = self.sm.forward(self.att)

        #dropout

        if self.dropout:
            keep = np.random.rand(*self.att.shape) >= self.dropout
            self.att = self.att * keep / (1 - self.dropout)

        # cv is context_vector
        self.cv = self.att @ self.v
        self.cv = np.transpose(self.cv, (0, 2, 1, 3))
        self.cv = self.cv.reshape(batches, self.context_window, self.d_out)
        self.cv = self.cv @ self.wo

        return self.cv

if __name__ == "__main__":

    rng = np.random.default_rng(12)

    cfg = {
        "d": 0.2,
        "batches": 8,
        "num_heads": 4,
        "emb": 16,
        "con": 8,
    }

    batch = rng.normal(0, 1, size = (cfg["batches"], cfg["con"], cfg["emb"]))

    att = MultiHeadAttention(dropout=cfg["d"], num_heads=cfg["num_heads"], embedding_size=cfg["emb"], context_window=cfg["con"], seed=0)

    cv = att.forward(batch=batch)

    print(cv)

    