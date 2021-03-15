import pandas as pd
import numpy as np

from task1 import model_evaluation
from task1.feature_exaction import Ngram, Tfidf
from task1.load_data import load_training_data


class LogisticClassifier:
    def __init__(self, alpha=0.1):
        self.N = 0  # 数据记录数（行数）
        self.D = 0  # 特征数
        self.w = np.zeros(self.D)
        self.alpha = alpha

    def fit(self, X, y):
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.w = np.zeros(self.D)

        for i in range(300):  # 迭代固定次数
            yy = self.predict_prob(X)[:, 1]
            yy = np.array(y - yy)
            # print(X.T.shape)
            # print(len(yy))
            tmp1 = X.T * yy
            tmp2 = sum(tmp1.T)
            self.w = self.w + self.alpha * 1 / self.N * sum((X.T * yy).T)

    def predict_prob(self, X):
        prob = []
        for i in range(len(X)):
            prob.append(1 / (1 + np.exp(-np.dot(X[i].reshape(-1), self.w))))
        prob = np.array([prob, prob]).T
        prob[:, 0] = 1 - prob[:, 0]
        return prob

    def predict(self, X):
        prob = self.predict_prob(X)
        res = []
        for i in range(len(prob)):
            if prob[i][0] < prob[i][1]:
                res.append(1)
            else:
                res.append(0)
        return np.array(res)


def one_hot(seq):
    N = len(seq)
    C = len(set(seq))
    res = np.zeros((N, C))
    for i in range(N):
        res[i][seq[i]] = 1
    return res


class SoftmaxClassifier:
    def __init__(self, alpha=0.1):
        self.N = 0  # 数据记录数（行数）
        self.D = 0  # 特征数
        self.C = 0  # 类别数
        self.w = np.zeros((self.D, self.C))  # 参数的维度是 DxC
        self.alpha = alpha

    def softmax(self, W, x):  # W是参数矩阵，x是输入向量
        assert (W.shape[0] == x.shape[0])  # W 为 DxC
        res = np.zeros(W.shape[1])
        tot = 0.0
        for i in range(W.shape[1]):
            res[i] = np.exp(W[:, i].dot(x))
            tot += res[i]
        res /= tot
        return res

    def fit(self, X, y):
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.C = len(set(y))
        self.w = np.zeros((self.D, self.C))  # 参数的维度是 DxC

        y = np.array(y)
        y_true = one_hot(y.reshape(-1))
        for i in range(1000):  # 迭代固定次数
            yy = np.array(y_true - self.predict_prob(X))
            tot = np.zeros((self.D, self.C))
            for i in range(self.N):
                tot = tot + X[i, :].reshape((-1, 1)) * yy[i, :].reshape((1, -1))
            self.w = self.w + self.alpha / self.N * tot

    def predict_prob(self, X):
        prob = np.zeros((X.shape[0], self.C))
        for i in range(X.shape[0]):
            prob[i, :] = self.softmax(self.w, X[i, :])
        return prob

    def predict(self, X):
        prob = self.predict_prob(X)
        res = np.zeros(len(X))
        for i in range(len(X)):
            idx = 0
            for j in range(1, prob.shape[1]):
                if prob[i][j] > prob[i][idx]:
                    idx = j
            res[i] = idx
        return res


documents, y = load_training_data()
X = Tfidf().fit_transform(documents)
# X = Ngram(2).fit_transform(documents)
lrm = SoftmaxClassifier()
lrm.fit(X, y)
y_predict = lrm.predict(X)
print(model_evaluation.accuracy_rate(y, y_predict))
