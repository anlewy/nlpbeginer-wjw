import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num=1, dropout_rate=0.5):
        super(LSTM, self).__init__()
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        if layer_num == 1:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layer_num, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layer_num,
                                dropout=dropout_rate, batch_first=True)
        self.init_weights()

    def init_weights(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_normal_(p)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.layer_num, batch_size, self.hidden_size),
                weight.new_zeros(self.layer_num, batch_size, self.hidden_size))

    def forward(self, x, x_lens, hidden):
        x_packed = nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True)
        output_packed, (h, c) = self.lstm(x_packed, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)
        return output, (h, c)


class LSTM_LM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size=128, dropout_rate=0.2, layer_num=1):
        super(LSTM_LM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = LSTM(embed_size, hidden_size, layer_num, dropout_rate)
        self.full = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.embed.weight)
        nn.init.xavier_normal_(self.full.weight)

    def forward(self, x, x_lens, hidden):
        """
        :param x: (batch, seq_len, input_size)
        :param x_lens: (batch, ), in descending order
        :param hidden: tuple(h,c), each has shape (num_layer, batch, hidden_size)
        :return: output: (batch, seq_len, hidden_size)
                tuple(h,c): each has shape (num_layer, batch, hidden_size)
        """
        x_embed = self.embed(x)
        hidden, (h, c) = self.lstm(self.dropout(x_embed), x_lens, hidden)
        out = self.full(self.dropout(hidden))
        return out, (h, c)

