import numpy as np
import pandas as pd
from keras.utils import Sequence


class SegSequence(Sequence):
    def __init__(self, csvfile: str, batch_size: int = None, oh: bool = True):
        self.df = pd.read_csv(csvfile)

        self.X_cols = [k for k in self.df.keys()][1:]
        self.y_cols = ['CLASS']

        self.batch_size = batch_size if batch_size is not None else len(self.df)
        self.oh = oh

    def __len__(self):
        return int(np.floor(len(self.df)/self.batch_size))

    def __getitem__(self, item):

        batch_input = np.zeros((self.batch_size, len(self.X_cols)))

        if self.oh:
            batch_output = np.zeros((self.batch_size, 7))
        else:
            batch_output = np.zeros((self.batch_size, 1))

        for bidx in range(0, self.batch_size):

            sidx = item*self.batch_size + bidx

            sample = self.df.iloc[sidx]

            X = sample[self.X_cols].values
            y = sample[self.y_cols].values[0]

            class_map = {
                'GRASS': 0,
                'FOLIAGE': 1,
                'WINDOW': 2,
                'PATH': 3,
                'BRICKFACE': 4,
                'CEMENT': 5,
                'SKY': 6
            }

            def one_hot(cls):
                idx = class_map[cls]

                oh = np.zeros((7, ))
                oh[idx] = 1

                return oh

            batch_input[bidx] = X

            if self.oh:
                batch_output[bidx] = one_hot(str(y))
            else:
                batch_output[bidx] = class_map[str(y)]

        return batch_input, batch_output


if __name__ == '__main__':
    seq = SegSequence('data/train.csv')

    bin, bout = seq.__getitem__(0)
    for i in range(0, bin.shape[0]):
        print("--%d--" % i)
        print(bin[i])
        print(bout[i])
