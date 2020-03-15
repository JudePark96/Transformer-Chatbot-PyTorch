"""
Author: Jude Park <judepark@kookmin.ac.kr>
"""


from konlpy.tag import Mecab
from utils.build_vocab import Vocabulary
from utils.data_utils import load_data


import torch as T
import pytest


class TestDataset(object):
    @pytest.fixture
    def seqs(self):
        mecab = Mecab()
        return mecab.morphs('1지망 학교 떨어졌어')

    @pytest.fixture
    def vocab(self):
        train, valid, train_y, valid_y, corpus = load_data('./rsc/data/chatbot_korean.csv')
        vocab = Vocabulary(corpus=corpus)
        vocab.build_vocab()
        return vocab

    def test_seq_max_len_with_pad(self, seqs, vocab):
        max_len = 128
        seq_len = len(seqs)
        seqs = T.LongTensor([vocab.get_token2idx(word) for word in seqs])
        seq_tensor = T.ones((max_len)).long()
        seq_tensor[:seq_len] = seqs

        for idx in seq_tensor[5:]:
            assert idx.item() == 1

        for i in range(seq_len):
            assert seqs[i].item() == seq_tensor[i]
