import torch.nn as nn
import torch
from transformers import BertTokenizer, BertModel


class COMBINED_MODEL(nn.Module):
    def __init__(self, num_labels, dropout_rate=0.1):
        super(COMBINED_MODEL, self).__init__()
        self.embed_size = 768
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.full = nn.Linear(self.embed_size, num_labels)
        self.dropout = nn.Dropout(dropout_rate)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.full.weight)
        self.full.weight.data.mul_(0.01)

    def forward(self, x):
        x_tokenized = self.tokenizer(x, return_tensors='pt', padding=True)
        x_bert = self.bert(**x_tokenized)
        x_embed = x_bert.pooler_output
        output = self.full(self.dropout(x_embed))
        return torch.softmax(output, 1)
