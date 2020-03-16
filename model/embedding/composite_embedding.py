"""
Author: Jude Park <judepark@kookmin.ac.kr>
"""


from model.embedding.positional_embedding import PositionalEmbedding
from model.embedding.token_embedding import TokenEmbedding
from torch.nn import Module

import torch as T


class CombineEmbedding(Module):
    def __init__(self, args: dict) -> None:
        super(CombineEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(args)
        self.pos_embedding = PositionalEmbedding(args)

    def forward(self, x) -> T.Tensor:
        return self.token_embedding(x) + self.pos_embedding(x)
