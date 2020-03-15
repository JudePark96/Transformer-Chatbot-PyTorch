"""
Author: Jude Park <judepark@kookmin.ac.kr>
"""

import torch.nn as nn
import pytest
import torch


class TestEmbedding(object):
    @pytest.fixture
    def seqs(self):
        # bs * seq_len
        return torch.LongTensor(320).random_(0, 320).view((32, 10))

    @pytest.fixture
    def lstm(self):
        return nn.LSTM(input_size=100, hidden_size=256, num_layers=1,
                       bidirectional=True, batch_first=True)

    @pytest.fixture
    def embed(self):
        return nn.Embedding(num_embeddings=350, embedding_dim=100)

    def test_seqs_shape(self, seqs):
        assert str(seqs.shape) == str('torch.Size([32, 10])')

    def test_embed_seqs_shape(self, seqs, embed):
        res = embed(seqs)
        assert str(res.shape) == str('torch.Size([32, 10, 100])')

    def test_lstm_shape(self, seqs, embed, lstm):
        output, (hidden, cell) = lstm(embed(seqs))
        # num_layers * 2, batch_size, hidden_size
        assert str(hidden.shape) == str('torch.Size([2, 32, 256])')
        # num_layers * 2, batch_size, hidden_size
        assert str(cell.shape) == str('torch.Size([2, 32, 256])')
        assert str(torch.cat((hidden[-1], hidden[-2]),
                             dim=1).shape) == str('torch.Size([32, 512])')