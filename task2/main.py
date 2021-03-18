from task2.data import get_data_iter
from task2.model import COMBINED_MODEL
from task2.train import train, evaluate
import torch.nn as nn
import torch.optim as optim

DATA_PATH = 'data'
BATCH_SIZE = 32
EMBED_SIZE = 64
HIDDEN_SIZE = 32
DROPOUT_RATE = 0.3
LAYER_NUM = 2
LEARNING_RATE = 0.01
EPOCHS = 3
DEVICE = 'cpu'
PATIENCE = 3
CLIP = 3

# 获取数据
train_iter, valid_iter, test_iter, TEXT, LABEL = get_data_iter(DATA_PATH, BATCH_SIZE)

# 三要素
model = COMBINED_MODEL(len(TEXT.vocab), EMBED_SIZE, HIDDEN_SIZE, len(LABEL.vocab), DROPOUT_RATE, LAYER_NUM)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 模型训练和模型评价
train(model, loss_func, optimizer, train_iter, valid_iter, EPOCHS, PATIENCE, CLIP)
acc = evaluate(model, loss_func, test_iter)
print("accuracy in test data is {}".format(acc))
