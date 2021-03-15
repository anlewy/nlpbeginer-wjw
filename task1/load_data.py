import pandas as pd
import numpy as np


def load_data():
    data = pd.read_csv("data/train.tsv", delimiter='\t')

    data = data.sample(frac=0.01)
    # print("size of data = ", len(data))
    texts = data['Phrase']
    labels = data['Sentiment']
    return texts, np.array(labels)
