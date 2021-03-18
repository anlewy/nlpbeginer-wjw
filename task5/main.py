import torch
import torch.nn as nn
import torch.optim as optim
from task5.data import get_data_iter
from task5.model import LSTM_LM
from task5.train import train, evaluate, generate

EPOCHS = 256
BATCH_SIZE = 16
EMBED_SIZE = 64
MOMENTUM = 0.9
LEARNING_RATE = 0.01

train_iter, valid_iter, test_iter, TEXT = get_data_iter()
pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
eos_idx = TEXT.vocab.stoi[TEXT.eos_token]

# 三要素
model = LSTM_LM(len(TEXT.vocab), EMBED_SIZE)
loss_func = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

train(model, loss_func, optimizer, train_iter, valid_iter, epochs=EPOCHS)
evaluate(model, loss_func, test_iter, is_dev=False)
try:
    while True:
        word = input("输入第一个字，或按Ctrl+C退出: ")
        generate(model, TEXT, eos_idx, word.strip())
except:
    pass
