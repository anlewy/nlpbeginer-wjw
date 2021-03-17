import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.1, layer_num=1):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        if layer_num == 1:
            self.net = nn.LSTM(input_size, hidden_size//2, layer_num, batch_first=True, bidirectional=True)
        else:
            self.net = nn.LSTM(input_size, hidden_size//2, layer_num, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.init_weights()

    def init_weights(self):
        for p in self.net.parameters():
            if p.dim() > 1:
                nn.init.normal_(p)
                p.data.mul_(0.01)
            else:
                p.data.zero_()
                p.data[self.hidden_size//2: self.hidden_size] = 1

    def forward(self, seqs, seq_lens):
        ordered_lens, index = seq_lens.sort(descending=True)
        ordered_seqs = seqs[index]

        packed_seqs = nn.utils.rnn.pack_padded_sequence(ordered_seqs, ordered_lens, batch_first=True)
        packed_output, _ = self.net(packed_seqs)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        recover_index = index.argsort()
        recover_output = output[recover_index]
        return recover_output


class ESIM(nn.Module):
    def __init__(self, vocab_size, num_labels, embed_size, hidden_size, dropout_rate=0.1, layer_num=1):
        super(ESIM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.bilstm1 = BiLSTM(embed_size, hidden_size, dropout_rate, layer_num)
        self.bilstm2 = BiLSTM(hidden_size, hidden_size, dropout_rate, layer_num)

        self.fullconn1 = nn.Linear(4*hidden_size, hidden_size)
        self.fullconn2 = nn.Linear(4*hidden_size, hidden_size)
        self.fullconn3 = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout_rate)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.embed.weight)
        self.embed.weight.data.mul_(0.01)
        nn.init.normal_(self.fullconn1.weight)
        self.fullconn1.weight.data.mul_(0.01)
        nn.init.normal_(self.fullconn2.weight)
        self.fullconn2.weight.data.mul_(0.01)
        nn.init.normal_(self.fullconn3.weight)
        self.fullconn3.weight.data.mul_(0.01)

    def soft_align_attention(self, seqs1, seq_lens1, seqs2, seq_lens2):
        pass

    def composition(self, seqs, seq_lens):
        pass

    def forward(self, seqs1, seq_lens1, seqs2, seq_lens2):
        pass