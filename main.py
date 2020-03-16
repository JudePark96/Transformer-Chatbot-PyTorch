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
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)

    args = parser.parse_args()
    data_path = args.data_path
    output_path = args.output_path

    train, valid, train_y, valid_y, corpus = load_data(data_path)
    vocab = Vocabulary(corpus)
    vocab.build_vocab()

    train_loader = get_loader(train, train_y, vocab, 64, 32, True)
    valid_loader = get_loader(valid, valid_y, vocab, 64, 32, True)
    model_args = get_base_config()

    model = Net(model_args).to(get_device_setting())
    optimizer = optim.Adam(params=model.parameters(), lr=model_args['lr'])
    trainer = Trainer(model_args, vocab, model, optimizer, output_path)

    print('********** trainer object has been initiated **********')
    print(model)
    print('********** trainer object has been initiated **********')

    trainer.train(10, train_loader, valid_loader)
