"""
Author: Jude Park <judepark@kookmin.ac.kr>
"""


from typing import Dict, List

from konlpy.tag import Mecab
from tqdm import tqdm


class Vocabulary(object):
    words: List[str]
    seqs: List[str]
    idx: int
    idx2token: Dict[int, str]
    token2idx: Dict[str, int]

    def __init__(self, corpus: list):
        self.corpus = corpus
        self.token2idx = {}
        self.idx2token = {}
        self.idx = 0
        self.seqs = []
        self.words = []

        # Mecab is default option.
        self.tokenizer = Mecab()
        self.special_tokens = [
            '<unk>',
            '<pad>',
        ]

        # initiate token dictionary, when initiated vocabulary class.
        self.init_token()

    def get_token2idx(self, token: str) -> int:
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        else:
            return self.token2idx[token]

    def get_idx2token(self, idx) -> str:
        if idx not in self.idx2token:
            # oov issue, so return <unk>
            return self.idx2token[0]
        else:
            return self.idx2token[idx]

    def get_vocab_size(self) -> int:
        if len(self.token2idx) == 0:
            raise ValueError('vocabulary has not been built.')
        return len(self.token2idx)

    def get_seqs(self) -> None:
        self.seqs = [self.tokenizer.morphs(
            seq.strip().lower()) for seq in tqdm(self.corpus)]

    def get_words(self) -> None:
        if len(self.seqs) == 0:
            self.get_seqs()

        words = [word for seq in self.seqs for word in seq]
        self.words = list(set(words))

    def init_token(self) -> None:
        for special_token in self.special_tokens:
            self.add_token(special_token)

    def add_token(self, token) -> None:
        if token not in self.token2idx:
            self.token2idx[token] = self.idx
            self.idx2token[self.idx] = token
            self.idx += 1

    def build_vocab(self) -> None:
        if len(self.words) == 0:
            self.get_words()

        for word in self.words:
            self.add_token(word)
