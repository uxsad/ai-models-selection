#!/usr/bin/env python3


import argparse
import gc
import logging
import math
import multiprocessing as mp
import pathlib
import time
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd

KEYS_TO_INCLUDE = {
    "middle.url",
    "middle.url.category",
    'middle.user_id'
}

KEYS_TO_IGNORE = {
    "middle.emotions.exists",  # Always True by design
    # The time of the middle object has served its purpose in the creation of
    # the interval and is now useless
    "middle.timestamp",
    # Often contains untreatable values (NaN or Inf) by design
    "middle.trajectory.slope",
}
KEYS_TO_PREDICT = {
    "middle.emotions.joy", "middle.emotions.fear", "middle.emotions.disgust",
    "middle.emotions.sadness", "middle.emotions.anger",
    "middle.emotions.valence", "middle.emotions.surprise",
    "middle.emotions.contempt", "middle.emotions.engagement"
}

logger = logging.getLogger(__name__)
gc.enable()


def range_type(x: str) -> float:
    x = float(x)
    if not 0 <= x <= 1:
        raise argparse.ArgumentTypeError(f"{x} is not in [0, 1]")
    return x


def setup_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Preprocess the dataset',
        epilog='Copyright (C) Andrea Esposito 2020.'
               ' Released under the GNU GPL v3 License.'
    )
    parser.add_argument(
        'dataset',
        help='The original dataset path.',
    )
    parser.add_argument(
        '--random', '-r',
        type=int,
        help='The random seed.'
    )
    parser.add_argument(
        '--jobs', '-j',
        type=int,
        help='The number of parallel jobs to use.',
        default=1,
    )
    parser.add_argument(
        '--size', '-s',
        type=range_type,
        default=1,
        help='The relative size of the final dataset.'
             ' Default is 1 (the entire dataset).'
    )
    parser.add_argument(
        '--discrete', '-d',
        dest='discrete_steps',
        default=7,
        type=int,
        help='The number of steps into which the emotions'
             ' will be discretized.'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Increase verbosity level.'
    )
    parser.add_argument(
        '--out', '-o',
        help='The output folder.',
        default=None,
        type=str
    )
    return parser


def merge_dataset(base_path: str = '.', width: int = None,
                  location: Optional[str] = None):
    """Merge the datasets' multiple files in single files."""

    base_path = pathlib.Path(base_path)

    # Checking preconditions
    if not base_path.exists():
        raise ValueError(f"The base path '{base_path}' does not exist")
    if not (base_path / 'users.csv').exists():
        raise FileNotFoundError("The base path does not contain the"
                                " 'users.csv' file, that contains the"
                                " users' data")
    if not (base_path / 'websites.csv').exists():
        raise FileNotFoundError("The base path does not contain the"
                                " 'websites.csv' file, that contains the"
                                " websites' data")

    # Load the users' data
    users = pd.read_csv(
        base_path / 'users.csv',
        index_col='id',
        dtype={
            'age': np.float32,
            'internet': np.float32,
            'gender': pd.StringDtype()
        },
        engine='c'
    )
    users['gender'] = pd.Categorical(
        users['gender'],
        categories={'m', 'f', 'a'}
    )
    users['age'] = pd.Categorical(users['age'], categories=range(6))

    # Load the websites' data
    websites = pd.read_csv(
        base_path / 'websites.csv',
        dtype={
            'count': np.float32,
            'category': pd.StringDtype()
        },
        engine='c'
    )
    websites['category'] = pd.Categorical(websites['category'])
    websites['url'] = pd.Categorical(websites['url'])

    def can_take_column(col: str) -> bool:
        if width is None and location is None:
            return col in KEYS_TO_INCLUDE | KEYS_TO_PREDICT or \
                   col not in KEYS_TO_IGNORE
        if width is not None and location is None:
            return col in KEYS_TO_INCLUDE | KEYS_TO_PREDICT or \
                   col not in KEYS_TO_IGNORE and col.startswith(f"{width}.")
        if width is None and location is not None:
            return col in KEYS_TO_INCLUDE | KEYS_TO_PREDICT or \
                   col not in KEYS_TO_IGNORE and f".{location}." in col
        if width is not None and location is not None:
            return col in KEYS_TO_INCLUDE | KEYS_TO_PREDICT or \
                   col not in KEYS_TO_IGNORE and \
                   col.startswith(f"{width}.{location}.")

    def get_users_data():
        for i, user_id in enumerate(users.index, 1):
            logger.info("Loading user '%s' (%d of %d)", user_id, i,
                        users.shape[0])
            path = base_path / user_id / 'aggregate.csv'
            if not path.exists():
                logger.warning("The file 'aggregate.csv' for the user '%s' was"
                               " not found.", user_id)
                yield pd.DataFrame()
            else:
                logger.debug("Loading CSV for user '%s'", user_id)
                df = pd.read_csv(
                    path,
                    usecols=can_take_column,
                    encoding='utf-8',
                    engine='c'
                )
                gc.collect()
                yield df

    logging.info("Reading the data from multiple CSV files")
    start_time = time.time()
    df = pd.concat(get_users_data())
    end_time = time.time()
    logger.info("Completed loading in %.3f seconds", end_time - start_time)
    logging.info(f"Loaded {df.shape[0]} objects")

    return df


def discretize_emotions(data: pd.DataFrame, steps: int = 7) -> pd.DataFrame:
    """Discretize the emotions.
    :param data: The dataframe containing the emotions to be discretized.
    :param steps: The number of steps into which the emotions' range will be
        discretized.
    :return: The discretized dataframe.
    """

    def get_step(val, minimum=0, maximum=100):
        step_width = (maximum - minimum) / steps
        for i in range(0, steps):
            if minimum + step_width * i <= val < minimum + step_width * (
                    i + 1):
                return i
        if val == maximum:
            return steps - 1
        return None

    df = data.copy(deep=True)
    df['middle.emotions.valence'] = df['middle.emotions.valence'].apply(
        lambda x: get_step(x, minimum=-100, maximum=100)
    )
    for k in KEYS_TO_PREDICT - {'middle.emotions.valence'}:
        df[k] = df[k].apply(get_step)

    return df


def get_value(data: pd.DataFrame, to_take: int, emotion: str, i: int,
              random_state: int = None):
    logging.debug(
        "Selecting from %s with value %d",
        emotion.split('.')[2], i
    )
    df = data.loc[data[emotion] == i]
    n = to_take if to_take < df.shape[0] else df.shape[0]
    logging.debug(
        "Taking %d objects for %s (with value %d)",
        n, emotion.split('.')[2], i
    )
    return df.sample(n=n, random_state=random_state)


def sample_dataset(df: pd.DataFrame, out_folder: str, split: float,
                   random_state: int = None, steps: int = 7,
                   n_jobs: int = 1):
    # Checking preconditions
    if split < 0 or split > 1:
        raise ValueError("The split value must be in [0, 1]")
    elif split == 1:
        return df
    elif split == 0:
        return pd.DataFrame(columns=df.columns)

    total_objects = math.ceil(df.shape[0] * split)
    to_take = math.ceil(total_objects / steps)

    out_path = pathlib.Path(out_folder) / f'aggregate-{split * 100}percent/'
    out_path.mkdir(parents=True, exist_ok=True)

    for emotion in KEYS_TO_PREDICT:
        func = partial(get_value, df, to_take, emotion,
                       random_state=random_state)
        if n_jobs == 1:
            final = []
            for i in range(steps):
                final.append(func(i))
        else:
            logging.info("Instantiating %d parallel processes", n_jobs)
            with mp.Pool(processes=n_jobs) as pool:
                final = pool.map(func, range(steps))
        logging.info(
            "Merging and saving output for %s",
            emotion.split('.')[2]
        )
        df = pd.concat(final)
        df.to_csv(
            out_path / f"{emotion.split('.')[2]}.csv",
            encoding='utf-8',
            index=False
        )
        logging.info("Took %d objects for %s", df.shape[0],
                     emotion.split('.')[2])

    print(f"Sampled dataset saved to '{out_path}'")


def main(*args):
    cli = setup_cli()
    args = cli.parse_args(*args)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.CRITICAL
    )
    dataset = merge_dataset(
        base_path=args.dataset
    )
    dataset = discretize_emotions(
        dataset,
        steps=args.discrete_steps
    )
    gc.collect()
    sample_dataset(
        dataset,
        args.out or args.dataset,
        split=args.size,
        steps=args.discrete_steps,
        n_jobs=args.jobs,
        random_state=args.random,
    )


if __name__ == '__main__':
    main()
