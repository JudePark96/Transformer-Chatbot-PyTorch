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
        'embedding_size': 100,
        'dropout_rate': 0.25,
        'hidden_size': 256,
        'num_layers': 6,
        'pad_idx':1,
        'lr': 1e5,
    })