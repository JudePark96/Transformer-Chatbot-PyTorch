"""
Author: Jude Park <judepark@kookmin.ac.kr>
"""


from typing import Union, Tuple

from sklearn.model_selection import train_test_split

import pandas as pd


def load_data(file_path: str) -> Tuple[list, list, list, list, list]:
    if file_path.split('/')[-1].split('.')[-1] != 'csv':
        raise ValueError('invalid file path')

    df = pd.read_csv(file_path, header=0)

    q, a = list(df['Q']), list(df['A'])

    train, valid, train_y, valid_y = train_test_split(q, a, test_size=0.10, random_state=42)

    corpus = q + a

    return train, valid, train_y, valid_y, corpus
