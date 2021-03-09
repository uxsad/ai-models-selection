from typing import Tuple

import pandas as pd


def split_dataset(data: pd.DataFrame, labels: pd.Series, frac: float = 0.7) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = data.sample(frac=frac, random_state=0)
    test = data.drop(train.index)

    train_labels = labels.loc[train.index]
    test_labels = labels.drop(train.index)

    return train, train_labels, test, test_labels
