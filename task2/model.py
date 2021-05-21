import torch.nn as nn
import torch
import torch.nn.functional as F


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num=1, dropout_rate=0.2):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        if layer_num == 1:
            self.lstm = nn.LSTM(input_size, hidden_size // 2, num_layers=layer_num, batch_first=True,
                                bidirectional=True)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size // 2, num_layers=layer_num, dropout=dropout_rate,
                                batch_first=True, bidirectional=True)
        self.init_weights()

    def init_weights(self):
        for p in self.lstm.parameters():
            if p.dim() > 1:
                nn.init.normal_(p)
                p.data.mul_(0.01)
            else:
                p.data.zero_()
                # TODO
                p.data[self.hidden_size // 2: self.hidden_size] = 1

    def forward(self, x, x_lens):
        x_packed = nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True)
        output_packed, _ = self.lstm(x_packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)
        return output


class COMBINED_MODEL(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_labels, dropout_rate=0.3, layer_num=1):
        super(COMBINED_MODEL, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = BiLSTM(embed_size, hidden_size, layer_num, dropout_rate)
        self.full = nn.Linear(2*hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout_rate)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.embed.weight)
        self.embed.weight.data.mul_(0.01)
        nn.init.normal_(self.full.weight)
        self.full.weight.data.mul_(0.01)

    def composition(self, x):
        p_avg = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p_max = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        return torch.cat([p_avg, p_max], 1)

    def forward(self, x, x_lens):
        x_embed = self.embed(x)
        x_hidden = self.lstm(self.dropout(x_embed), x_lens)
        x_com = self.composition(self.dropout(x_hidden))
        output = self.full(self.dropout(x_com))
        return torch.softmax(output, 1)
