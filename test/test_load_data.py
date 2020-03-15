"""
Author: Jude Park <judepark@kookmin.ac.kr>
"""

from konlpy.tag import Mecab

import pandas as pd
import pytest


class TestLoadData(object):
    @pytest.fixture
    def file_path(self):
        return './rsc/data/chatbot_korean.csv'

    @pytest.fixture
    def target_file_type(self):
        return 'csv'

    @pytest.fixture
    def mecab(self):
        return Mecab()

    def test_target_path_is_csv(self, file_path:str, target_file_type: str) -> None:
        assert file_path.split('/')[-1].split('.')[-1] == target_file_type

    def test_load_corpus(self, file_path: str):
        if file_path.split('/')[-1].split('.')[-1] != 'csv':
            raise ValueError('invalid file path')

        df = pd.read_csv(file_path, header=0)

        q, a = list(df['Q']), list(df['A'])

        assert a[0] == '하루가 또 가네요.'
        assert q[0] == '12시 땡!'

    def test_mecab_seq(self, mecab: Mecab):
        assert mecab.morphs('하루가 또 가네요') == ['하루', '가', '또', '가', '네요']

    def test_corpus_to_flatten(self):
        data = [['12', '시', '땡', '!'], ['1', '지망', '학교', '떨어졌', '어']]
        flatten = [word for seq in data for word in seq]
        assert len(flatten) == 9
