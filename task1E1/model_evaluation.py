def accuracy_rate(y_true, y_predict):
    assert len(y_true) == len(y_predict), '两列表长度需一致'
    return sum(y_true == y_predict) / len(y_true)


def error_rate(y_true, y_predict):
    return 1 - accuracy_rate(y_true, y_predict)
