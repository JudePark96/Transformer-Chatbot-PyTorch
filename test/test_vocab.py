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

    def test_sos_check(self, vocab):
        assert vocab.idx2token[2] == '<sos>'

    def test_eos_check(self, vocab):
        assert vocab.idx2token[3] == '<eos>'

    def test_equals_idx2token_token2idx(self, vocab):
        idxs = [2, 4, 6, 8]
        tokens = [vocab.get_idx2token(idx) for idx in idxs]
        idxs_ = [vocab.get_token2idx(token) for token in tokens]

        assert idxs == idxs_

    def test_unregistered_token_return_unk(self, vocab):
        assert vocab.get_token2idx(token='내이름은이효리거꾸로해도이효리') == 0

    def test_vocab_size(self, vocab):
        assert vocab.get_vocab_size() == 6854
