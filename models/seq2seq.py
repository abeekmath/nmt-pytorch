import torch
import random
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers=1, drop_prob=0.3) -> None:
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=drop_prob)

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=False,
                            dropout=drop_prob)

    def forward(self, src):

        embedding = self.dropout(self.embedding(src))
        output, (hidden, cell) = self.dropout(embedding)
        return output, hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers=1, drop_prob=0.3) -> None:
        super(Decoder, self).__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=drop_prob)

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=False,
                            dropout=drop_prob)
        self.fc_out = nn.Linear(hidden_size, input_size)

    def forward(self, input, hidden, cell):

        embedding = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.lstm(embedding)
        prediction = self.fc_out(output)
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, enc, dec, device):
        super(Seq2Seq, self).__init__()

        self.encoder = enc
        self.decoder = dec
        self.device = device

        assert self.encoder.hidden_size == self.decoder.hidden_size, "hidden sizes must be equal"
        assert self.encoder.num_layers == self.decoder.num_layers, "num_layers must be equal"

    def forward(self, src, trg, teacher_force_ratio=0.5):

        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        output_vocab_size = self.decoder.input_size

        outputs = torch.zeros(trg_len, batch_size, output_vocab_size)
        output, hidden, cell = self.encoder(src)
        input = None

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_force_ratio
            input = trg[t] if teacher_force else output

        return outputs


if __name__ == '__main__':

    # unit test pending
    pass
