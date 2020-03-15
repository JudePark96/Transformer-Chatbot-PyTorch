"""
Author: Jude Park <judepark@kookmin.ac.kr>
"""

from model.embedding.positional_embedding import PositionalEmbedding

import pytest
import torch


class TestEmbedding(object):
    @pytest.fixture
    def seqs(self):
        # [bs x seq_len]
        return torch.LongTensor(320).random_(0, 320).view((32, 10))

    @pytest.fixture
    def args(self):
        return {'max_len': 128, 'd_model':300}

    @pytest.fixture
    def embed(self, args):
        return PositionalEmbedding(args)

    def test_position_embed_shape(self, seqs, embed):
        o = embed(seqs)
        assert str(o.shape) == 'torch.Size([1, 10, 300])'

