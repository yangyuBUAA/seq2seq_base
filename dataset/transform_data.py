def transform_data():
    in_path_label = "/Users/yangyu/PycharmProjects/seq2seq_base/dataset/couplet/train/in.txt.label"
    out_path_label = "/Users/yangyu/PycharmProjects/seq2seq_base/dataset/couplet/train/out.txt.label"
    vocab_path_label = "/Users/yangyu/PycharmProjects/seq2seq_base/dataset/couplet/vocabs.label"

    with open(vocab_path_label, "r", encoding="utf-8") as vocab_label_r:
        vocabs2index = {vocab.strip():index+1 for index, vocab in enumerate(vocab_label_r.readlines())}

    print(vocabs2index)

    with open(in_path_label, "r", encoding="utf-8") as in_label_r:
        in_datas = [i.strip() for i in in_label_r.readlines()]
        in_datas_digit = list()
        for line in in_datas:
            line_digit = " ".join([str(vocabs2index[vocab]) for vocab in line.split()])
            in_datas_digit.append(line_digit)

    with open(in_path_label+".digit", "w", encoding="utf-8") as digit_w:
        for line in in_datas_digit:
            digit_w.write(line+"\n")

    with open(out_path_label, "r", encoding="utf-8") as out_label_r:
        out_datas = [i.strip() for i in out_label_r.readlines()]
        out_datas_digit = list()
        for line in out_datas:
            line_digit = " ".join([str(vocabs2index[vocab]) for vocab in line.split()])
            out_datas_digit.append(line_digit)

    with open(out_path_label+".digit", "w", encoding="utf-8") as digit_w:
        for line in out_datas_digit:
            digit_w.write(line+"\n")


if __name__ == '__main__':
    transform_data()
