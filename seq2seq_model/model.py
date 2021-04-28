import torch
import torch.nn as nn

import random


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, lstm_hiddensize):
        super(Encoder, self).__init__()
        self.lstm_hiddensize = lstm_hiddensize
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, lstm_hiddensize, bidirectional=True, batch_first=False)

    def forward(self, input_sequence):
        """
        :param input_sequence: shape of input_sequence: (batch_size, max_seq_length)
        :return:
        """
        embedded = self.embedding(input_sequence)  # shape of embedded: (batch_size, max_seq_length, embedding_size)
        embedded = embedded.permute(1, 0, 2)  # shape of embedded: (max_seq_len, batch_size, embedding_size)
        output, (hidden, cell) = self.lstm(embedded)
        # shape of output: (max_seq_len, batch_size, lstm_hiddensize)
        # shape of hidden and cell: (num_layers*num_directions, batch_size, lstm_hiddensize)
        # print(hidden.shape, cell.shape)
        return hidden, cell


class Decoder(nn.Module):
    """
    decoder为一个单词一个单词做输入
    """
    def __init__(self, target_vocab_size, embedding_size, lstm_hiddensize):
        super(Decoder, self).__init__()
        self.lstm_hiddensize = lstm_hiddensize
        self.embedding = nn.Embedding(target_vocab_size, embedding_size)
        self.decoder_lstm = nn.LSTM(embedding_size, lstm_hiddensize, bidirectional=True, batch_first=False)
        # shape of self.decoder_lstm: 1, batch_size, lstm_hiddensize
        self.linear_layer = nn.Sequential(
            nn.Linear(2*lstm_hiddensize, lstm_hiddensize),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(lstm_hiddensize, target_vocab_size)
        )

    def forward(self, w, hidden, cell):
        """
        每次迭代传入上一个time step的解码输出以及上一个time step
        :param w: shape of w: (batch_size, 1)
        :param hidden: shape of hidden: (num_directions*num_layers, batch_size, lstm_hiddensize)
        :param cell: shape of cell: (num_directions*num_layers, batch_size, lstm_hiddensize)
        :return:
        """
        embedded = self.embedding(w)  # shape of embeded: (batch_size, 1, embedding_dim)
        # print(embedded.shape)
        embedded = embedded.permute(1, 0, 2)  # shape of embedded: (1, batch_size, embedding_dim)
        # print(embedded.shape)
        lstm_out, (hidden, cell) = self.decoder_lstm(embedded, (hidden, cell))
        # shape of lstm_out (1, batch_size, lstm_hiddensize)
        # shape of hidden and cell: (num_directions*num_layers, batch_size, lstm_hiddensize)
        lstm_out = lstm_out.squeeze()  # shape of lstm_out (batch_size, lstm_hiddensize)
        # print(lstm_out.shape)
        classifier_result = self.linear_layer(lstm_out)
        # shape of classifier_result: (batch_size, target_vocab_size)
        return classifier_result, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_size, lstm_hiddensize):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_vocab_size, embedding_size, lstm_hiddensize)
        self.decoder = Decoder(output_vocab_size, embedding_size, lstm_hiddensize)
        # self.teacher_ratio = teacher_ratio
        self.trg_vocab_size = output_vocab_size
        assert self.encoder.lstm_hiddensize == self.decoder.lstm_hiddensize

    def forward(self, src, trg, teacher_ratio):
        """
        :param x: shape of src (batch_size, max_seq_len)
        :param y: shape of trg (batch_size, max_seq_len_output)
        :return:
        """
        trg_len = len(trg[0])
        batch_size = len(trg)
        hidden, cell = self.encoder(src)
        # print("shape of hidden and cell{}".format(hidden.shape))
        # shape of hidden and cell (num_directions*num_layers, batch_size, lstm_hiddensize)
        input_0 = trg[:, 0].unsqueeze(-1)  # shape of input_0 (batch_size, 1)
        # print("shape of input_0{}".format(input_0.shape))
        # print(input_0.shape)
        output, hidden, cell = self.decoder(input_0, hidden, cell)
        # print("shape of output{}".format(output.shape))
        # shape of output: (batch_size, trg_vocab_size)
        # shape of hidden: (num_directions*num_layers, batch_size, lstm_hiddensize)
        # shape of cell: (num_directions*num_layers, batch_size, lstm_hiddensize)

        outputs = torch.zeros(trg_len, batch_size, self.trg_vocab_size)
        # print("shape of outputs:{}".format(outputs.shape))
        outputs[0] = output
        for i in range(1, trg_len):
            # print(i)
            teacher_force = random.random() < teacher_ratio
            if teacher_force:
                inputs = trg[:, i].unsqueeze(-1)
                # print("inputs.shape:{}".format(inputs.shape))
                # print(type(hidden), type(cell))
                outputs[i], hidden, cell = self.decoder(inputs, hidden, cell)
                # print(outputs[i].shape)
            else:
                # print(type(hidden), type(cell))
                inputs = outputs[i-1]  # shape of inputs: (batch_size, trg_vocab_size)
                # print("inputs.shape{}".format(inputs.shape))
                inputs = torch.argmax(inputs, dim=1).unsqueeze(-1)  # shape of inputs: (batch_size, 1)
                # print("inputs.shape{}".format(inputs.shape))
                outputs[i], hidden, cell = self.decoder(inputs, hidden, cell)
        # shape of outputs: (trg_len, batch_size, self.trg_vocab_size)
        outputs = outputs.permute(1, 0, 2)
        # print(outputs.shape)
        # shape of outputs: (batch_size, trg_len, vocab_size)
        # print(outputs)
        # print(outputs.shape)
        return outputs


if __name__ == '__main__':
    a = torch.tensor([[1, 6, 4, 2, 1, 0, 0, 0, 0, 0],
         [8, 6, 8, 5, 9, 0, 0, 0, 0, 0],
         [5, 2, 1, 7, 6, 0, 0, 0, 0, 0]])

    b = torch.tensor([[4, 2, 6, 8, 4, 3, 0, 0, 0, 0],
         [5, 34, 5, 7, 3, 1, 0, 0, 0, 0],
         [0, 7, 4, 6, 3, 2, 0, 0, 0, 0]])

    seq2seq = Seq2Seq(9134, 9134, 300, 100)
    print(seq2seq(a, b, 0.8).shape)