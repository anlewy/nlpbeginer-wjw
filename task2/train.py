import torch

from task2.data import get_data_iter

data_train_iter, data_test_iter, num_inputs, num_outputs = get_data_iter()


class LinearNet(torch.nn.Module):
    def __init__(self, num_in, num_out):
        super(LinearNet, self).__init__()
        self.linear = torch.nn.Linear(num_in, num_out)

    def forward(self, x):
        return self.linear(x.view(x.shape[0], -1))


net = LinearNet(num_inputs, num_outputs)
torch.nn.init.normal_(net.linear.weight, mean=0, std=0.01)
torch.nn.init.constant_(net.linear.bias, val=0)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)


def accuracy(_y_hat, y):
    return _y_hat.argmax(dim=1).float().mean().item()


def net_accuracy(data_iter, net):
    right_sum, n = 0.0, 0
    for X, y in data_iter:
        # 从迭代器data_iter中获取X和y
        right_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        # 计算准确判断的数量
        n += y.shape[0]
        # 通过shape[0]获取y的零维度（列）的元素数量
    return right_sum / n


num_epochs = 32
for epoch in range(num_epochs):
    train_l_sum, train_right_sum, n = 0.0, 0.0, 0

    for Xt, yt in data_train_iter:
        y_hat = net(Xt)
        l = loss(y_hat, yt).sum()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        train_l_sum += l.item()
        train_right_sum += (y_hat.argmax(dim=1) == yt).sum().item()
        n += yt.shape[0]
    test_acc = net_accuracy(data_test_iter, net)  # 测试集的准确率
    print('epoch %d, loss %.4f, train right %.3f, test acc %.3f' % (
        epoch + 1, train_l_sum / n, train_right_sum / n, test_acc))
