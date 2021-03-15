import pandas as pd
import numpy as np


def load_data(file, sep, frac=0.01):
    data = pd.read_csv(file, delimiter=sep)

    data = data.sample(frac=frac)
    # print("size of data = ", len(data))
    texts = data['Phrase']
    labels = data['Sentiment']
    return texts, np.array(labels)


def load_training_data(frac=0.01):
    return load_data("data/train.tsv", "\t", frac)


def load_testing_data(frac=0.01):
    return load_data("data/test.tsv", "\t", frac)
