import torch
from torchtext.legacy.data import Field, LabelField, Iterator, BucketIterator, TabularDataset


def get_data_iter(data_path="data", batch_size=32, device='cpu'):
    TEXT = Field(batch_first=True, include_lengths=True, lower=True)
    LABEL = LabelField(batch_first=True, include_lengths=True)

    fields = {
        'Phrase': ('phrase', TEXT),
        'Sentiment': ('sentiment', LABEL)
    }

    train_valid_data, test_data = TabularDataset.splits(
        path=data_path,
        train='train.tsv',
        test='test.tsv',
        format='tsv',
        fields=fields
    )

    train_data, valid_data = train_valid_data.splits([0.8, 0.2])

    TEXT.build_vocab(train_data)
    LABEL.build_vocab(valid_data)

    train_iter, valid_iter = BucketIterator.splits(
        (train_data, valid_data),
        batch_sizes=(batch_size, batch_size),
        device=device,
        sort_key=lambda e: len(e.phrase),
        sort_within_batch=True,
        repeat=False,
        shuffle=True
    )

    test_iter = Iterator(
        test_data,
        batch_size=batch_size,
        device=device,
        sort=False,
        sort_within_batch=False,
        repeat=False,
        shuffle=False
    )

    return train_iter, valid_iter, test_iter, TEXT, LABEL
