import pandas as pd


def split_dataset(data, labels):
    train = data.sample(frac=0.7, random_state=0)
    test = data.drop(train.index)

    train_labels = labels.loc[train.index]
    test_labels = labels.drop(train.index)

    return train, train_labels, test, test_labels
