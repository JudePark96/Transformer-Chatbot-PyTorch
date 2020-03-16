"""
Author: Jude Park <judepark@kookmin.ac.kr>
"""


from model.embedding.composite_embedding import CombineEmbedding
from config.base_experiment_config import get_device_setting
from torch.nn.modules.transformer import Transformer
from torch.nn import Module


import torch.nn as nn
import torch as T


class Net(Module):
    def __init__(self, args: dict) -> None:
        super().__init__()
        self.args = args
        self.transformer = Transformer(
            d_model=args['d_model'],
            nhead=args['nhead'],
            num_encoder_layers=args['num_encoder_layers'],
            num_decoder_layers=args['num_decoder_layers'],
            dim_feedforward=args['dim_feedforward']
        )

        self.embedding = CombineEmbedding(args=args)
        self.projection = nn.Linear(
            args['d_model'], args['vocab_size'])

    def forward(self, question: T.Tensor, answer: T.Tensor) -> T.Tensor:
        """
        :param question: [bs x seq_len]
        :param answer: [bs x seq_len]
        :return:
        """
        q_embed = self.embedding(question.long())
        a_embed = self.embedding(answer.long())

        q_mask = question == self.args['pad_idx']
        a_mask = answer == self.args['pad_idx']
        mem_q_mask = q_mask.clone()
        tgt_mask = self.transformer.generate_square_subsequent_mask(
            answer.size(1))

        # [seq x bs x dim] -> [bs x seq x dim]
        q_embed = T.einsum('ijk->jik', q_embed)
        a_embed = T.einsum('ijk->jik', a_embed)

        attn = self.transformer(src=q_embed,
                                tgt=a_embed,
                                src_key_padding_mask=q_mask,
                                tgt_key_padding_mask=a_mask,
                                memory_key_padding_mask=mem_q_mask,
                                tgt_mask=tgt_mask.to(get_device_setting()))

        attn = T.einsum('ijk->jik', attn)
        logits = self.projection(attn)
        print('logits shape ', logits.shape)

        return logits
