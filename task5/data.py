from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import pandas as pd


def preprocess(filename, max_length=128):
    with open(filename, encoding='utf-8') as f:
        poetries = []
        poetry = []
        for line in f:
            tmp = line.strip()
            if tmp == '' or len(poetry) + len(tmp) > max_length:
                poetries.append(poetry)
                poetry = []
            poetry.extend(tmp)
        pd.DataFrame([' '.join(poetry) for poetry in poetries], columns=['texts']).to_csv("data/poetry.csv")


def get_data_iter(batch_size=16, device='cpu', eos_token='[EOS]'):
    preprocess("data/poetryFromTang.txt")
    TEXT = Field(eos_token=eos_token, batch_first=True, include_lengths=True)
    fields = {
        'texts': ('text', TEXT)
    }

    all_data = TabularDataset.splits(path='data', train='poetry.csv', format='csv', fields=fields)[0]
    train_data, valid_data, test_data = all_data.split([0.8, 0.1, 0.1])
    TEXT.build_vocab(train_data)

    train_iter, valid_iter, test_iter = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_sizes=(batch_size, batch_size, batch_size),
        device=device,
        sort_key=lambda x : len(x.text),
        sort_within_batch=True,
        repeat=False,
        shuffle=True
    )

    return train_iter, valid_iter, test_iter, TEXT
