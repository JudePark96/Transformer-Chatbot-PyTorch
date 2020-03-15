"""
Author: Jude Park <judepark@kookmin.ac.kr>
"""


from torch.nn import Embedding, Module


import torch as T


class TokenEmbedding(Module):
    def __init__(self, args: dict) -> None:
        super().__init__()
        self.token_embed = Embedding(
            args['vocab_size'], args['embed_dim'], padding_idx=args['pad_idx'])

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.token_embed(x)  # [bs x seq_len x embed_dim]
