import torch

from task1.feature_exaction import Tfidf
from task1.load_data import load_testing_data, load_training_data


def get_data_iterator():
    documents, y = load_training_data(0.1)
    n = len(y)
    train_num = n * 2 // 3
    tfidfM = Tfidf()

    documents_train, y_train = documents[:train_num], y[:train_num]
    X_train = tfidfM.fit_transform(documents_train)
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    data_train_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=10,
                                                  shuffle=True)
    documents_test, y_test = documents[train_num:], y[train_num:]
    X_test = tfidfM.transform(documents_test)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)
    data_test_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=10,
                                                 shuffle=True)
    print("size of train_data:{}, size of test_data:{}".format(len(y_train), len(y_test)))
    return data_train_iter, data_test_iter, tfidfM.feature_num, len(set(y))
