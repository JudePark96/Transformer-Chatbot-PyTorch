"""
Author: Jude Park <judepark@kookmin.ac.kr>
PyTorch Transformer Test Class
"""


from utils.build_vocab import Vocabulary
from utils.data_utils import load_data


import pytest


class TestVocab(object):
    @pytest.fixture
    def vocab(self):
        train, valid, train_y, valid_y, corpus = load_data('./rsc/data/chatbot_korean.csv')
        vocab = Vocabulary(corpus=corpus)
        vocab.build_vocab()

        return vocab

    def test_vocab_size_is_valid(self, vocab):
        assert len(vocab.idx2token) == len(vocab.token2idx)
        assert len(vocab.idx2token) == len(vocab.words) + len(vocab.special_tokens)

    def test_unk_check(self, vocab):
        assert vocab.idx2token[0] == '<unk>'

    def test_pad_check(self, vocab):
        assert vocab.idx2token[1] == '<pad>'

    def test_equals_idx2token_token2idx(self, vocab):
        idxs = [2, 4, 6, 8]
        tokens = [vocab.get_idx2token(idx) for idx in idxs]
        idxs_ = [vocab.get_token2idx(token) for token in tokens]

        assert idxs == idxs_
