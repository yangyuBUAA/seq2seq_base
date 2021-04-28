def label_data():
    in_path = "/Users/yangyu/PycharmProjects/seq2seq_base/dataset/couplet/train/in.txt"
    out_path = "/Users/yangyu/PycharmProjects/seq2seq_base/dataset/couplet/train/out.txt"
    vocab_path = "/Users/yangyu/PycharmProjects/seq2seq_base/dataset/couplet/vocabs"

    with open(in_path, "r", encoding="utf-8") as in_r:
        lines = in_r.readlines()
        for index, line in enumerate(lines):
            lines[index] = "<GO> " + line.strip() + " <STOP>"

    with open(in_path+".label", "w", encoding="utf-8") as in_w:
        for line in lines:
            in_w.write(line + "\n")

    with open(out_path, "r", encoding="utf-8") as out_r:
        lines = out_r.readlines()
        for index, line in enumerate(lines):
            lines[index] = "<GO> " + line.strip() + " <STOP>"

    with open(out_path+".label", "w", encoding="utf-8") as out_w:
        for line in lines:
            out_w.write(line + "\n")

    with open(vocab_path, "r", encoding="utf-8") as vocab_r:
        vocabs = vocab_r.readlines()
        for index, vocab in enumerate(vocabs):
            vocabs[index] = vocab.strip()
        print(vocabs)

    vocabs.append("<GO>")
    vocabs.append("<STOP>")

    with open(vocab_path+".label", "w", encoding="utf-8") as vocab_w:
        for vocab in vocabs:
            vocab_w.write(vocab + "\n")

if __name__ == '__main__':
    label_data()
