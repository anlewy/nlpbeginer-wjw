import torch

from task1.feature_exaction import Tfidf
from task1.load_data import load_data

documents, y = load_data()
X = Tfidf().fit_transform(documents)
X = torch.tensor(X)
y = torch.tensor(y)
data_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=10, shuffle=True)

num_inputs = X.shape[1]
num_outputs = len(set(y))


class LinearNet(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = torch.nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        return y


net = LinearNet(num_inputs, num_outputs)
torch.nn.init.normal_(net.linear.weight, mean=0, std=0.01)
torch.nn.init.constant_(net.linear.bias, val=0)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)


def accuracy(y_hat, y):
    return y_hat.argmax(dim=1).float().mean().item()


def net_accurary(data_iter, net):
    right_sum, n = 0.0, 0
    for X, y in data_iter:
        # 从迭代器data_iter中获取X和y
        right_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        # 计算准确判断的数量
        n += y.shape[0]
        # 通过shape[0]获取y的零维度（列）的元素数量
    return right_sum / n


num_epochs = 1024
for epoch in range(num_epochs):
    train_l_sum, train_right_sum, n = 0.0, 0.0, 0

    for Xt, yt in data_iter:
        y_hat = net(Xt)
        l = loss(y_hat, yt).sum()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        train_l_sum += l.item()
        train_right_sum += (y_hat.argmax(dim=1) == yt).sum().item()
        n += yt.shape[0]
    test_acc = net_accurary(data_iter, net)  # 测试集的准确率
    print('epoch %d, loss %.4f, train right %.3f, test acc %.3f' % (
        epoch + 1, train_l_sum / n, train_right_sum / n, test_acc))
