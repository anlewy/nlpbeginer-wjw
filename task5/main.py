import torch
import torch.nn as nn
import torch.optim as optim
from task5.data import get_data_iter
from task5.model import LSTM_LM


BATCH_SIZE = 16
EMBED_SIZE = 64
MOMENTUM = 0.9
LEARNING_RATE = 0.01


train_iter, valid_iter, test_iter, TEXT = get_data_iter()
pad_idx = TEXT.vocab.stoi(TEXT.pad_token)
eos_idx = TEXT.vocab.stoi(TEXT.eos_token)

model = LSTM_LM(len(TEXT.vocab), EMBED_SIZE)
loss_func = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
