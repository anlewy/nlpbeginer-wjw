#part 1 - bert feature base
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import torch
import transformers as tfs
import warnings

warnings.filterwarnings('ignore')

train_df = pd.read_csv('data/SST2_train.tsv', delimiter='\t', header=None)

train_set = train_df[:3000]   #取其中的3000条数据作为我们的数据集
print("Train set shape:", train_set.shape)
train_set[1].value_counts()   #查看数据集中标签的分布

model_class, tokenizer_class, pretrained_weights = (tfs.BertModel, tfs.BertTokenizer, 'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

#add_special_tokens 表示在句子的首尾添加[CLS]和[END]符号
train_tokenized = train_set[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

train_max_len = 0
for i in train_tokenized.values:
    if len(i) > train_max_len:
        train_max_len = len(i)

train_padded = np.array([i + [0] * (train_max_len-len(i)) for i in train_tokenized.values])
print("train set shape:",train_padded.shape)

#output：train set shape: (3000, 66)

train_attention_mask = np.where(train_padded != 0, 1, 0)

train_input_ids = torch.tensor(train_padded).long()
train_attention_mask = torch.tensor(train_attention_mask).long()
with torch.no_grad():
    train_last_hidden_states = model(train_input_ids, attention_mask=train_attention_mask)

train_last_hidden_states[0].size()

train_features = train_last_hidden_states[0][:,0,:].numpy()
train_labels = train_set[1]

train_features, test_features, train_labels, test_labels = train_test_split(train_features, train_labels)

lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)
lr_clf.score(test_features, test_labels)