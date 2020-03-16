"""
Author: Jude Park <judepark@kookmin.ac.kr>
"""


from config.base_experiment_config import get_base_config
from utils.build_vocab import Vocabulary
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.nn import Module
from tqdm import tqdm



class Trainer(object):

    def __init__(self, args: dict, vocab: Vocabulary, model: Module) -> None:
        super().__init__()

        if vocab.get_vocab_size() == args['vocab_size']:
            raise ValueError('vocabulary has not been initiated.')

        self.args = args
        self.vocab = vocab
        self.model = model

    def train(self, epoch: int, loader: DataLoader) -> None:
        # define what loss function is
        loss_fn = CrossEntropyLoss(ignore_index=self.args['pad_idx'])

        for ep_iter in tqdm(range(1, epoch + 1)):
            for i, (question, answer) in tqdm(loader):
                pass
            pass
        pass

    def evaluate(self, loader: DataLoader):
        pass
