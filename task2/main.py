from task2.data import get_data_iter
import torch
import torch.nn as nn
import torch.optim as optim

DATA_PATH = 'data'
BATCH_SIZE = 32
EMBED_SIZE = 64
HIDDEN_SIZE = 32
DROPOUT_RATE = 0.3
LAY_NUM = 2
LEARNING_RATE = 0.01
EPOCHS = 16
DEVICE = 'cpu'

# 获取数据
train_iter, valid_iter, test_iter, TEXT, LABEL = get_data_iter(data_path=DATA_PATH, batch_size=BATCH_SIZE, device=DEVICE)

# 三要素
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam()

# 模型训练和模型评价
