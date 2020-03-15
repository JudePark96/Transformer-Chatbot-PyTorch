"""
Author: Jude Park <judepark@kookmin.ac.kr>
"""


from torch.utils.data import DataLoader, Dataset
from utils.build_vocab import Vocabulary
from konlpy.tag import Mecab


import torch as T

from utils.data_utils import load_data


class ConversationDataset(Dataset):
    def __init__(self, question: list, answer: list, vocab: Vocabulary, max_len: int = 128):
        self.question = question
        self.answer = answer
        self.vocab = vocab
        self.max_len = max_len
        self.tokenizer = Mecab()

    def __getitem__(self, idx: int):
        q_tokenized = self.tokenizer.morphs(self.question[idx])
        a_tokenized = self.tokenizer.morphs(self.answer[idx])

        q_len = len(q_tokenized)
        a_len = len(a_tokenized)

        q = T.ones(self.max_len).long()
        a = T.ones(self.max_len).long()

        q_tensor = T.LongTensor([self.vocab.get_token2idx(word)
                                 for word in self.tokenizer.morphs(self.question[idx])])

        a_tensor = T.LongTensor([self.vocab.get_token2idx(word)
                                 for word in self.tokenizer.morphs(self.answer[idx])])

        q[:q_len] = q_tensor
        a[:a_len] = a_tensor

        return q, a

    def __len__(self) -> int:
        assert len(self.question) == len(self.answer)
        return len(self.question)


def get_loader(question: list,
               answer: list,
               vocab: Vocabulary,
               max_len:int,
               batch_size: int,
               shuffle: bool) -> DataLoader:
    dataset = ConversationDataset(question, answer, vocab, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == '__main__':
    train, valid, train_y, valid_y, corpus = load_data('../rsc/data/chatbot_korean.csv')
    vocab = Vocabulary(corpus)
    vocab.build_vocab()
    loader = get_loader(train, train_y, vocab, 128, 32, True)
    for i, (question, answer) in enumerate(loader):
        print(question, answer)
        break