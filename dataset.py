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

def get_value(data: pd.DataFrame, to_take: int, emotion: str, i: int,
              random: int = None):
    logging.info("Taking the rows of the dataset having '%s' to %d",
                 emotion, i)
    df = data.loc[data[emotion] == i]
    logging.info("Taken %d rows, over an accepted maximum of %d",
                 df.shape[0], to_take)
    n = to_take if to_take < df.shape[0] else df.shape[0]
    logging.info("Selecting %d random rows", n)
    return df.sample(n=n, random_state=random)


@click.command()
@click.argument("dataset",
                type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.argument("size", type=click.FloatRange(0, 1))
@click.argument("classes", type=click.IntRange(1, None))
@click.option("-o", "--output", type=click.Path(dir_okay=True, file_okay=False))
@click.option("-r", "--random", type=click.INT)
@click.option("-j", "--jobs", type=click.IntRange(1, None))
def stratify(dataset, size, classes, output, random=None, jobs=1):
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting the stratification")
    df = pd.read_csv(dataset, engine='c')
    logging.info("Loading the dataset: DONE")
    elements_by_class = math.ceil((df.shape[0] * size) / classes)
    logging.info("Taking %d elements by class, i.e. ceil(%d * %.3f / %d)",
                 elements_by_class, df.shape[0], size, classes)
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
        logging.info("Saving %d rows for '%s'",
                     df.shape[0], emotion.split('.')[2])
        df.to_csv(
            out_path / f"{emotion.split('.')[2]}.csv",
            encoding='utf-8',
            index=False
        )
        logging.info("-" * 80)


if __name__ == '__main__':
    stratify()
