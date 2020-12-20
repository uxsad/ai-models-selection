import pandas as pd
import pathlib
import numpy as np

KEYS_TO_INCLUDE = {
    "middle.url", "middle.url.category", "middle.id", 'middle.user_id'
}
KEYS_TO_PREDICT = {
    "middle.emotions.joy", "middle.emotions.fear", "middle.emotions.disgust",
    "middle.emotions.sadness", "middle.emotions.anger",
    "middle.emotions.valence", "middle.emotions.surprise",
    "middle.emotions.contempt", "middle.emotions.engagement"
}
KEYS_TO_IGNORE = {
    "middle.emotions.exists",  # Always True by design
    # The time of the middle object has served its purpose in the
    # creation of
    # the interval and is now useless
    "middle.timestamp",
    # Often contains untreatable values (NaN or Inf) by
    # design
    "middle.trajectory.slope",
}


def load(path, emotion, width=None, location=None):
    """Load the dataset."""
    def can_take_column(col: str) -> bool:
        if col in KEYS_TO_IGNORE:
            return False
        if col in KEYS_TO_INCLUDE | KEYS_TO_PREDICT:
            return True

        if not width and not location:
            return True
        if width and not location:
            return col.startswith(f"{width}.")
        if not width and location:
            return f".{location}." in col

        return col.startswith(f"{width}.{location}.")

    target = f"middle.emotions.{emotion}"
    if target not in KEYS_TO_PREDICT:
        raise ValueError("{emotion} is not a supported emotion")

    users = pd.read_csv(path / 'users.csv',
                        index_col='id',
                        dtype={
                            'age': np.float32,
                            'internet': np.float32,
                            'gender': pd.StringDtype()
                        })
    users['gender'] = pd.Categorical(users['gender'],
                                     categories={'m', 'f', 'a'})
    users['age'] = pd.Categorical(users['age'], categories=range(6))

    websites = pd.read_csv(path / 'websites.csv',
                           dtype={
                               'count': np.float32,
                               'category': pd.StringDtype()
                           })
    websites['category'] = pd.Categorical(websites['category'])
    websites['url'] = pd.Categorical(websites['url'])

    df = pd.read_csv(
        path / f'{emotion}.csv',
        engine='c',
        usecols=can_take_column,
        encoding='utf-8',
        index_col="middle.id",
    )
    df['user.age'] = df['middle.user_id'].map(users['age'])
    df['user.internet'] = df['middle.user_id'].map(users['internet'])
    df['user.gender'] = df['middle.user_id'].map(users['gender'])
    # The user id is no longer needed
    df.drop(columns=['middle.user_id'], inplace=True)

    # OneHot encoder
    df['user.gender'] = df['user.gender'].map({
        'm': 'male',
        'f': 'female',
        'a': 'other'
    })
    df = pd.get_dummies(df, prefix_sep='.', columns=['user.gender'])
    # Ordinal encoder
    df['middle.url'] = df['middle.url'].map(
        {u: i
         for i, u in enumerate(pd.Categorical(df['middle.url']))})
    # Ordinal encoder
    df['middle.url.category'] = df['middle.url.category'].map({
        c: i
        for i, c in enumerate(pd.Categorical(df['middle.url.category']))
    })

    # Filling NaN values on target columns: by design if a value isn't there it
    # means that it was under 1 and can then be approximated to 0 (see
    # https://shorturl.at/JOTU5 and https://shorturl.at/etM09).
    x = df.drop(columns=KEYS_TO_PREDICT)
    y = df[KEYS_TO_PREDICT].fillna(0)

    return x, y[target]
