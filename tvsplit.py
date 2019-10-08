import pandas as pd
import numpy as np
import plac


@plac.annotations(
    datafile=('Path to the data file', 'option', 'i', str),
    train=('Percentage of training samples', 'option', 't', float),
    val=('Percentage of validation samples', 'option', 'v', float),
)
def main(datafile: str = 'data/data.csv',
         train: float = 0.7,
         val: float = 0.2):

    df = pd.read_csv(datafile)

    train_n = int(np.round(train*len(df)))
    val_n = int(np.round(val*len(df)))
    test_n = len(df) - (train_n + val_n)

    samples_all = df.sample(len(df))

    samples_train = samples_all[:train_n]
    samples_val = samples_all[train_n:train_n+val_n]

    samples_train.to_csv('data/train.csv', index=False)
    samples_val.to_csv('data/val.csv', index=False)

    if test_n > 0:
        samples_test = samples_all[train_n + val_n:]
        samples_test.to_csv('data/test.csv', index=False)


if __name__ == '__main__':
    plac.call(main)