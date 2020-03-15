from torch.nn import Module


import torch as T
import math


class PositionalEmbedding(Module):
    def __init__(self, args: dict):
        super().__init__()

        pe = T.zeros(args['max_len'], args['d_model']).float()
        pe.require_grad = False

        position = T.arange(0, args['max_len']).float().unsqueeze(1)
        div_term = (T.arange(0, args['d_model'], 2).float() * -(math.log(10000.0) / args['d_model'])).exp()

        pe[:, 0::2] = T.sin(position * div_term)
        pe[:, 1::2] = T.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x) -> T.Tensor:
        return self.pe[:, :x.size(1)]


if __name__ == '__main__':
    print(PositionalEmbedding({'max_len': 128, 'd_model': 300})(T.LongTensor(320).random_(0, 320).view((32, 10))).shape)