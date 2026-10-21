import numpy as np
from embedding import *

class GPT:

    def __init__(self, tokenizer, context_window, seed):

        self.tok_emb = Embedding(embedding_size=8, len=tokenizer.n_vocab, seed=seed)
        self.pos_emb = Embedding(embedding_size=8, len=context_window, seed=seed)

if __name__ == "__main__":
    import tiktoken

    seed=0

    tokenizer = tiktoken.get_encoding('gpt2')

    tr = GPT(tokenizer=tokenizer, context_window=128, seed=seed)

    sentence = [1, 5, 12, 8]
    print(tr.tok_emb.embedding[sentence, :])
