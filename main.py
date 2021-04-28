from dataset import couplet_dataset
from seq2seq_model.model import Seq2Seq

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import torch

def train():
    cop_dataset = couplet_dataset.Couplet_dataset()
    vocab_size = cop_dataset.vocab_nums
    seq2seq = Seq2Seq(vocab_size, vocab_size, 300, 100)

    dataloader = DataLoader(dataset=cop_dataset, batch_size=32, shuffle=True)
    lossf = CrossEntropyLoss()
    optimizer = Adam(seq2seq.parameters(), lr=0.01)
    seq2seq = seq2seq
    for epoch in range(100):
        for index, batch in enumerate(dataloader):
            batch_input = batch[:, :10]
            batch_trg = batch[:, 10:]
            optimizer.zero_grad()
            # shape of batch_input: (batch_size, max_seq_length)
            # shape of batch_trg: (batch_size, max_seq_length)
            output_head = seq2seq(batch_input, batch_trg, teacher_ratio=0.8)
            # shape of output_head: (batch_size, trg_len, vocab_size)
            output_head = output_head.permute(0, 2, 1)
            loss = lossf(output_head, batch_trg)
            # loss = 0
            # for i in range(len(batch_input[0])):
            #     loss = loss + lossf(output_head[:, i, :].squeeze(), batch_input[:, i])
            loss.backward()

            optimizer.step()
            print("epoch:{}, batch:{}, loss:{}".format(epoch+1, index+1, loss.data.item()))

    torch.save(seq2seq, "model.bin")

if __name__ == '__main__':
    train()