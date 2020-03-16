"""
Author: Jude Park <judepark@kookmin.ac.kr>
"""

from collections import OrderedDict


import torch as T


def get_device_setting():
    return T.device('cuda') if T.cuda.is_available() else T.device('cpu')


def get_base_config() -> OrderedDict:
    """
    baseline config
    :return:
    """
    return OrderedDict({
        'embed_dim': 512,
        'lr': 1e-4,
        'd_model':512,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'dim_feedforward': 2048,
        'dropout_rate': 0.25,
        'hidden_size': 256,
        'pad_idx':1,
        'vocab_size': 6854,
        'nhead':8,
        'max_len': 64,
    })