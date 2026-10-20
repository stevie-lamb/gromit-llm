"""
Here we want to define the dataset class, and also create a function that 
creates an iterable dataloader. One that loops over the text dataset and at
iteration supplies an input batch and a target batch. I suppose what we will have is a huge
folder full of txt files which we will want to loop through in training. Perhaps we just
pass a single file to this class then deal with the loop in the training file? Yeah, that
makes it easier to deal with here. This should also only concern tokenized data as
I will create my own tokenizer, and then where should the embedding go? It needs it's own class.
"""

import numpy as np

class TextDataset:
    """
    Note to self: the data loader should be able to iterate over inputs and targets.
    Each time we iterate it should be a batch of inputs? so each input is of dimension
    [embedding_size, context_size]

    Input:
    txt: np.ndarray; shape --> length of input?
    """
    def __init__(self, txt, context_window, stride): # text is tokenized text
        self.txt_tokenised = txt
        self.context_window = context_window
        self.stride = stride

        self.input_ids = []
        self.target_ids = []

        for i in range(len(txt) - 1):
            self.input_ids.append(txt[i])
            self.target_ids.append(txt[i + 1])

    def _batch(self):
        pass

    def create_dataloader(self, shuffle=False):
        pass



if __name__ == "__main__":


    txt = np.random.randint(0, 12000, 128)

    data = TextDataset(txt=txt, context_window=128, stride=128)

    print(data.input_ids)

