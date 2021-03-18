from task3.data import get_data_iter
from task3.model import ESIM
from task3.train import train, evaluate
import torch.nn as nn
import torch.optim as optim

DATA_PATH = 'data'
BATCH_SIZE = 32
DEVICE = 'cpu'
EMBED_SIZE = 32
HIDDEN_SIZE = 32
DROPOUT_RATE = 0.1
LAY_NUM = 2
LEARNING_RATE = 0.01
EPOCHS = 5

# 获取数据
train_iter, valid_iter, test_iter, TEXT, LABEL = get_data_iter(DATA_PATH, BATCH_SIZE, DEVICE)

# 三要素
model = ESIM(len(TEXT.vocab), len(LABEL.vocab), EMBED_SIZE, HIDDEN_SIZE, DROPOUT_RATE, LAY_NUM)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 模型训练和模型评价
train(model, loss_func, optimizer, train_iter, valid_iter, EPOCHS)
acc = evaluate(model, loss_func, test_iter)
print(acc)
