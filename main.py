"""
Author: Jude Park <judepark@kookmin.ac.kr>
"""


from config.base_experiment_config import get_base_config, get_device_setting
from utils.build_vocab import Vocabulary
from utils.data_utils import load_data
from utils.dataloder import get_loader
from model.transformer import Net
from trainer import Trainer


import torch.optim as optim


if __name__ == '__main__':
    train, valid, train_y, valid_y, corpus = load_data('./rsc/data/chatbot_korean.csv')
    vocab = Vocabulary(corpus)
    vocab.build_vocab()
    loader = get_loader(train, train_y, vocab, 64, 32, True)
    args = get_base_config()

    model = Net(args).to(get_device_setting())
    optimizer = optim.Adam(params=model.parameters(), lr=args['lr'])
    Trainer(args, vocab, model, optimizer).train(1, loader)