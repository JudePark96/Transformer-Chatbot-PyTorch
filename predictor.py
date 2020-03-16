"""
Author: Jude Park <judepark@kookmin.ac.kr>
Predictor do the inference by given trained model.
"""


from utils.build_vocab import Vocabulary
from torch.nn import Module


import torch as T


class Predictor(object):
    def __init__(self,
                 vocab: Vocabulary,
                 seqs: list[str],
                 model_path: str) -> None:
        self.vocab = vocab
        self.seqs = seqs
        self.model_path = model_path

    def inference(self):
        print(f'source seqs: {self.seqs}')

        model = T.load(self.model_path)
        model.eval()




        pass
    pass