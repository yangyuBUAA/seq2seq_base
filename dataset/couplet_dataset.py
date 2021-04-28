import torch
from torch.utils.data import Dataset

in_path_label = "/Users/yangyu/PycharmProjects/seq2seq_base/dataset/couplet/train/in.txt.label.digit"
out_path_label = "/Users/yangyu/PycharmProjects/seq2seq_base/dataset/couplet/train/out.txt.label.digit"
vocab_path_label = "/Users/yangyu/PycharmProjects/seq2seq_base/dataset/couplet/vocabs.label"

class Couplet_dataset(Dataset):
    def __init__(self):
        self.data = self.read_raw_data()  # shape of self.data: torch.tensor(data_sum, 20)
        self.vocab_nums = self.vocab()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def read_raw_data(self):
        with open(in_path_label, "r", encoding="utf-8") as in_read:
            ins = [line.strip().split() for line in in_read.readlines()]
        for index, line in enumerate(ins):
            if len(line) < 10:
                for i in range(10-len(line)):
                    ins[index].append(i)
            elif len(line) > 10:
                ins[index] = line[:10]
            else:
                pass
            ins[index] = [int(i) for i in ins[index]]

        ins = torch.tensor(ins)
        with open(out_path_label, "r", encoding="utf-8") as out_read:
            outs = [line.strip().split() for line in out_read.readlines()]
        for index, line in enumerate(outs):
            if len(line) < 10:
                for i in range(10-len(line)):
                    outs[index].append(i)
            elif len(line) > 10:
                outs[index] = line[:10]
            else:
                pass
            outs[index] = [int(i) for i in outs[index]]
        outs = torch.tensor(outs)
        # print(ins)
        # print(outs)
        # print(ins.shape, outs.shape)
        # print(torch.cat((ins, outs), 1).shape)
        return torch.cat((ins, outs), 1)

    def vocab(self):
        with open(vocab_path_label, "r", encoding="utf-8") as vocab_r:
            vocabs = vocab_r.readlines()
        return len(vocabs) + 1

if __name__ == '__main__':
    c = Couplet_dataset()
    print(c[0])
    print(len(c))