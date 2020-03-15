"""
Author: Jude Park <judepark@kookmin.ac.kr>
PyTorch Transformer Test Class
"""


import pytest
import torch as T
import torch.nn as nn


from torch.nn.modules.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer)


class TestTransformer(object):
    @pytest.fixture
    def rand_src(self) -> T.Tensor:
        return T.rand(10, 32, 512)  # [seq_lens x bs x d_model]

    @pytest.fixture
    def rand_tgt(self) -> T.Tensor:
        return T.randn(20, 32, 512)  # [seq_lens x bs x d_model]

    @pytest.fixture
    def encoder_layer(self) -> nn.Module:
        return TransformerEncoderLayer(d_model=512, nhead=8)

    @pytest.fixture
    def encoder(self, encoder_layer: nn.Module) -> nn.Module:
        return TransformerEncoder(encoder_layer, num_layers=6)

    @pytest.fixture
    def decoder_layer(self) -> nn.Module:
        return TransformerDecoderLayer(512, 8)

    @pytest.fixture
    def decoder(self, decoder_layer: nn.Module) -> nn.Module:
        return TransformerDecoder(decoder_layer, num_layers=6)

    def test_torch_version(self):
        assert T.__version__ == '1.4.0'

    def test_encoder(self, encoder: nn.Module, rand_src: T.Tensor) -> None:
        assert str(encoder(rand_src).shape) == 'torch.Size([10, 32, 512])'

    def test_decoder(self,
                     encoder: nn.Module,
                     decoder: nn.Module,
                     rand_src: T.Tensor,
                     rand_tgt: T.Tensor) -> None:
        decoder_output = decoder(rand_tgt, encoder(rand_src))
        assert (str(decoder_output.shape) == 'torch.Size([20, 32, 512])')
