#!/usr/bin/env python3


import argparse
import gc
import csv
import click
import logging
import p_tqdm
import tqdm
from pathos.multiprocessing import ProcessPool as Pool
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
    "middle.user_id"
}

KEYS_TO_IGNORE = {
    "middle.emotions.exists",  # Always True by design
    # The time of the middle object has served its purpose in the creation of
    # the interval and is now useless
    "middle.timestamp",
    # Often contains untreatable values (NaN or Inf) by design "middle.trajectory.slope",
}
KEYS_TO_PREDICT = {
    "middle.emotions.joy", "middle.emotions.fear", "middle.emotions.disgust",
    "middle.emotions.sadness", "middle.emotions.anger",
    "middle.emotions.valence", "middle.emotions.surprise",
    "middle.emotions.contempt", "middle.emotions.engagement"
}

logger = logging.getLogger(__name__)
gc.enable()


@click.group()
def main():
    pass


def get_step(val, steps, minimum=0, maximum=100):
    if not val:
        return 0
    val = float(val)
    step_width = (maximum - minimum) / steps
    for i in range(steps):
        if minimum + step_width * i <= val < minimum + step_width * (i + 1):
            return i
    if val >= maximum:
        return steps - 1
    return None


@main.command()
@click.argument("dataset",
                type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.argument("steps", type=int)
@click.option("-o", "--output", type=click.File('w'))
def merge(dataset, steps, output):
    dataset = pathlib.Path(dataset)
    if not (dataset / 'users.csv').exists():
        raise FileNotFoundError("The base path does not contain the"
                                " 'users.csv' file, that contains the"
                                " users' data")
    if not (dataset / 'websites.csv').exists():
        raise FileNotFoundError("The base path does not contain the"
                                " 'websites.csv' file, that contains the"
                                " websites' data")
    # Load the users' ids
    users = pd.read_csv(
        dataset / 'users.csv',
        index_col='id',
        usecols=['id'],
        engine='c'
    ).index

    def process_row(row):
        row['middle.emotions.valence'] = get_step(
            row['middle.emotions.valence'], steps, minimum=-100, maximum=100)
        for k in KEYS_TO_PREDICT - {'middle.emotions.valence'}:
            row[k] = get_step(row[k], steps)
        return row

    def get_user_interactions(path):
        if not path.exists():
            return []
        with open(path, "r") as dataset:
            csv_file = csv.DictReader(dataset)
            for row in csv_file:
                yield process_row(row)

    users_path = [dataset / user_id / 'aggregate.csv' for user_id in users]
    users_data = (get_user_interactions(path) for path in users_path)
    with open(users_path[1], 'r') as f:
        csv_file = csv.DictReader(f)
        writer = csv.DictWriter(output, next(csv_file).keys())
        writer.writeheader()
    for data in tqdm.tqdm(users_data, total=len(users_path), leave=False):
        for line in data:
            writer.writerow(line)


def get_value(data: pd.DataFrame, to_take: int, emotion: str, i: int,
              random: int = None):
    df = data.loc[data[emotion] == i]
    n = to_take if to_take < df.shape[0] else df.shape[0]
    return df.sample(n=n, random_state=random)


@main.command()
@click.argument("dataset",
                type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.argument("size", type=click.FloatRange(0, 1))
@click.argument("classes", type=click.IntRange(1, None))
@click.option("-o", "--output", type=click.Path(dir_okay=True, file_okay=False))
@click.option("-r", "--random", type=click.INT)
@click.option("-j", "--jobs", type=click.IntRange(1, None))
def stratify(dataset, size, classes, output, random=None, jobs=1):
    df = pd.read_csv(dataset, engine='c')
    elements_by_class = math.ceil((df.shape[0] * size) / classes)
    out_path = pathlib.Path(output) / f'aggregate-{size}/'
    out_path.mkdir(parents=True, exist_ok=True)
    for emotion in KEYS_TO_PREDICT:
        func = partial(get_value, df, elements_by_class, emotion, random=random)
        if jobs == 1:
            final = []
            for i in range(classes):
                final.append(func(i))
        else:
            logging.info("Instantiating %d parallel processes", jobs)
            final = p_tqdm.p_umap(func, range(classes), num_cpus=jobs)
            # with mp.Pool(processes=jobs) as pool:
            #     final = pool.map(func, range(classes))
        df = pd.concat(final)
        df.to_csv(
            out_path / f"{emotion.split('.')[2]}.csv",
            encoding='utf-8',
            index=False
        )


if __name__ == '__main__':
    main()
