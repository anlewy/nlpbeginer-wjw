from torchtext.legacy.data import Iterator, BucketIterator, TabularDataset, Field, LabelField


def get_data_iter(data_path="data", batch_size=32, device="cpu"):
    TEXT = Field(batch_first=True, include_lengths=True, lower=True)
    LABEL = LabelField(batch_first=True, include_lengths=False)

    fields = {
        'sentence1': ('premise', TEXT),
        'sentence2': ('hypothesis', TEXT),
        'gold_label': ('label', LABEL)
    }

    train_data, valid_data, test_data = TabularDataset.splits(
        path=data_path,
        train='snli_1.0_train.jsonl',
        validation='snli_1.0_dev.jsonl',
        test='snli_1.0_test.jsonl',
        format='json',
        fields=fields,
        filter_pred=lambda e: e.label != '-'  # 如果label是'-'，意味着没打标签
    )

    TEXT.build_vocab(train_data)
    LABEL.build_vocab(valid_data)

    train_iter, valid_iter = BucketIterator.splits(
        (train_data, valid_data),
        batch_sizes=(batch_size, batch_size),
        device=device,
        sort_key=lambda e: len(e.premise) + len(e.hypothesis),
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
