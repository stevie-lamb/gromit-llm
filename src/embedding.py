import numpy as np

class Embedding:
    def __init__(self, embedding_size, len, seed=0):

        self.emb_size = embedding_size
        self.len = len

        rng = np.random.default_rng(seed)

        self.embedding = rng.normal(0, 0.01, (self.len, self.emb_size))

    def get_embedding(self, token_id):
        """
        This function takes the id of a token and retrieves the appropriate embedding
        """

        return self.embedding[:, token_id]

    def _get_shape(self):
        """
        Retrieves embedding dimensions for helpfulness
        """
        x = f"The embedding size is: {self.emb_size}"
        y = f"The total number of tokens, and therefore model dimension is: {self.total_tokens}"
        z = f"The shape of the embedding is: {self.embedding.shape}"

        return (f"{x} \n{y} \n{z}  \n")

if __name__ == "__main__":
    pass