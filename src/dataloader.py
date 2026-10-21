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
import os 
import tiktoken

tokenizer = tiktoken.get_encoding('gpt2')

class TextDataset:
    """
    Note to self: the data loader should be able to iterate over inputs and targets.
    Each time we iterate it should be a batch of inputs? so each input is of dimension
    [embedding_size, context_size]

    Input:
    txt: np.ndarray; shape --> length of input?
    """
    def __init__(self, txt, tokenizer, context_window, stride):

        self.txt_tokenised = tokenizer.encode(txt)
        self.t_txt = np.array(self.txt_tokenised)
        self.context_window = context_window
        self.stride = stride

        # as list ------> defunct; arrays quicker?
        self.input_ids = np.array([], dtype=np.int64)
        self.target_ids = np.array([], dtype=np.int64)

        for i in range(0, len(self.txt_tokenised) - context_window + 1, self.stride):
            self.input_ids = np.append(self.input_ids, self.txt_tokenised[i: i+context_window])
            self.target_ids = np.append(self.target_ids, self.txt_tokenised[i+1:i+1+context_window])

        self.input_ids = self.input_ids.reshape(int(len(self.input_ids) / context_window), context_window)
        self.target_ids = self.target_ids.reshape(int(len(self.target_ids) / context_window), context_window)
        
        


        # This block ensures we can reshape to the correct number of tokens! Otherwise we would be left with a wonky number.
        #self.unused_tokens = len(self.t_txt) % context_window
        #self.t_txt = self.t_txt[-self.unused_tokens:]

        # for simplicity, let
        # self.stride = stride
        # self._arrays = round(self.context_window / self.stride, 0)



        # self._samples = ((len(self.t_txt)) // self.context_window) - 1

        # # self.input_arr = np.zeros(shape=(self._samples, self.context_window))

        # for i in range(0, self.context_window, self.stride):

        #     self.input_arr = self.t_txt[i: i + self._samples * (self.context_window)]
        #     self.input_arr = self.input_arr.reshape(self._samples, self.context_window)

        #     self.target_arr = self.t_txt[i + 1: i + (self._samples*self.context_window) +1]
        #     self.target_arr = self.target_arr.reshape(self._samples, self.context_window)

    def create_dataloader(self, batch_size, shuffle=False):
        """
        This function should return a list of tuples [in_batch_1, target_batch_1), ... , (in_batch_n, target_batch_n)]
        where each batch in the tuple is of shape [batch_size, context_window]
        """

        total_batches = int(len(self.input_ids) // batch_size)

        loader = []

        if shuffle:
            # if shuffle, then randomise the order of the rows in input_ids and target_ids, but so they 
            # are randomised equivalently, if that makes sense (they should still line up)
            pass

        for i in range(0, total_batches, batch_size):

            inputs = self.input_ids[i + batch_size, :]
            targets = self.target_ids[i + batch_size, :]

            loader.append((inputs, targets))

        return loader





if __name__ == "__main__":
    pass
    
