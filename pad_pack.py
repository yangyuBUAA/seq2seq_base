import torch


if __name__ == '__main__':
    a = torch.tensor([[1, 4, 7],
                      [1, 0, 0],
                      [2, 0, 0],
                      [3, 0, 0]])
    b = torch.nn.utils.rnn.pack_padded_sequence(a, [3, 1, 1, 1], batch_first=True)
    # print(b)

    c = torch.nn.utils.rnn.pad_packed_sequence(b, batch_first=True)
    # print(c)

    embedding = torch.nn.Embedding(10, 2)
    d = embedding(a)
    # print(d, d.shape)
    e = torch.nn.utils.rnn.pack_padded_sequence(d, [3, 1, 1, 1], batch_first=True)
    # print(e)
    lstm = torch.nn.LSTM(2, 10, batch_first=True)
    d_out = lstm(d)
    e_out = lstm(e)
    print(d_out[1][0].shape)
    print("---")
    print(torch.nn.utils.rnn.pad_packed_sequence(e_out[0], batch_first=True))
    print(e_out[1][0])