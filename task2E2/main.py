from task2E2.data import get_data_iter
from task2E2.model import COMBINED_MODEL
from task2E2.train import train, evaluate
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

DATA_PATH = 'data'
BATCH_SIZE = 32
EMBED_SIZE = 64
HIDDEN_SIZE = 32
DROPOUT_RATE = 0.3
LAYER_NUM = 2
LEARNING_RATE = 0.01
EPOCHS = 128
DEVICE = 'cpu'
PATIENCE = 3
CLIP = 3


class TextSet(Dataset):

    def __init__(self):
        self.frame = pd.read_csv('data/train.tsv', sep='\t')
        self.num_labels = len(set(self.frame['Sentiment']))

    def __getitem__(self, item):
        return self.frame.iloc[item]['Phrase'], self.frame.iloc[item]['Sentiment']

    def __len__(self):
        return len(self.frame)


# 获取数据
BATCH_SIZE = 32
texts = TextSet()
# texts_train, texts_valid, texts_test = texts.split([0.7, 0.15, 0.15])
textloader = DataLoader(texts, batch_size=BATCH_SIZE, shuffle=True)


# train_iter, valid_iter, test_iter, TEXT, LABEL = get_data_iter(DATA_PATH, BATCH_SIZE)

# 三要素
model = COMBINED_MODEL(texts.num_labels, DROPOUT_RATE)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 模型训练和模型评价
train(model, loss_func, optimizer, textloader, textloader, EPOCHS, PATIENCE, CLIP)
acc = evaluate(model, loss_func, textloader)
print("accuracy in test data is {}".format(acc))
