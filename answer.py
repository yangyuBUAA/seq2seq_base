import torch

from seq2seq_model.model import Seq2Seq

vocab_label_path = "/Users/yangyu/PycharmProjects/seq2seq_base/dataset/couplet/vocabs.label"

def sentence2input_tensor(sentence):
    sentence_list = list(sentence)
    sentence_list.insert(0, "<GO>")
    sentence_list.append("<STOP>")
    with open(vocab_label_path, "r", encoding="utf-8") as vocab_r:
        vocab2index = {vocab.strip():index+1 for index, vocab in enumerate(vocab_r.readlines())}
    sentence_digit = [vocab2index[i] for i in sentence_list]
    # print(sentence_digit)
    if len(sentence_digit) > 10:
        sentence_digit = sentence_digit[:10]
    elif len(sentence_digit) < 10:
        for i in range(10-len(sentence_digit)):
            sentence_digit.append(0)
    else:
        pass
    print(sentence_digit)

def output_tensor2sentence(output_tensor):
    # shape of output_tensor is (10, 1, vocab_size)
    print(output_tensor.shape)
    output_tensor = output_tensor.squeeze()
    # shape of output_tensor = (10, vocab_size)
    answer_digit_tensor = torch.argmax(output_tensor, dim=1)
    # answer_digit_tensor shape is (10, 1)
    answer_digit_list = list(answer_digit_tensor)
    print(answer_digit_list)


def answer(sentence):
    answer_robot = torch.load("model.bin")
    input_tensor = sentence2input_tensor(sentence)
    output = answer_robot(input_tensor, input_tensor, 0)
    answer_digit_list = output_tensor2sentence(output)
    print(answer_digit_list)

if __name__ == '__main__':
    # sentence2input_tensor("晚风摇树树还挺")
    answer("晚风摇树树还挺")