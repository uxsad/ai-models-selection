import itertools
import tqdm
import codecs
import functools
import json
import multiprocessing as mp
import pickle
import time
import logging

import mlpack
import numpy as np
import yaml
import signal
from .model import execute_model

logger = logging.getLogger(__name__)


def forward_feature_selection(data,
                              labels,
                              algorithm,
                              n_jobs=1,
                              show_progress=False):
    def initializer():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def get_col_set(features):
        to_add = (features + [i] for i in range(data[0].shape[1])
                  if i not in features)
        return ((data[0].iloc[:, sub], data[1].iloc[:, sub]) for sub in to_add)

    feature_set = []
    res = []
    g = functools.partial(execute_model,
                          labels=labels,
                          algorithm=algorithm)
    #  if n_jobs == 1:
    #  logger.info("Starting in serial mode")
    #  else:
    #  logger.info("Starting %d parallel jobs", n_jobs)
    start_time = time.time()
    try:
        r = tqdm.trange(data[0].shape[1], desc="Cycles",
                        leave=False) if show_progress else range(
                            data[0].shape[1])
        for _ in r:
            columns_set = get_col_set(feature_set)
            if show_progress:
                columns_set = tqdm.tqdm(columns_set,
                                        desc="Columns",
                                        total=data[0].shape[1] -
                                        len(feature_set),
                                        leave=False)
            #  if n_jobs == 1:
            #  results = map(g, columns_set)
            #  else:
            #  with mp.Pool(n_jobs, initializer=initializer) as pool:
            #  results = pool.map(g, columns_set, chunksize=(300))
            results = map(g, columns_set)
            best = max(results, key=lambda x: x[1])
            model, acc, training_time, cols = best
            pickled = codecs.encode(pickle.dumps(model, protocol=4),
                                    "base64").decode()
            feature_set = [data[0].columns.get_loc(col) for col in cols]
            res.append({
                'model': pickled,
                'columns': cols,
                'accuracy': {
                    'absolute': f'{acc[0]}/{len(labels[1])}',
                    'relative': acc[1]
                },
                'time': training_time  # in seconds
            })
    except KeyboardInterrupt:
        logger.warning("Received SIGINT! Stopping early")
    end_time = time.time()
    return res, end_time - start_time


def exhaustive_feature_selection(data,
                                 labels,
                                 algorithm,
                                 n_jobs=1,
                                 show_progress=False):
    def initializer():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    feature_set = itertools.chain.from_iterable(
        itertools.combinations(range(data[0].shape[1]), k + 1)
        for k in range(data[0].shape[1]))
    columns_set = ((data[0].iloc[:, list(sub)], data[1].iloc[:, list(sub)])
                   for sub in feature_set)

    res = []

    logger.info("Starting %d parallel jobs", n_jobs)
    start_time = time.time()
    with mp.Pool(n_jobs, initializer=initializer) as pool:
        g = functools.partial(execute_model,
                              labels=labels,
                              algorithm=algorithm)
        results = pool.imap_unordered(g, columns_set, chunksize=(300))

        if show_progress:
            results = tqdm.tqdm(results,
                                desc="Features",
                                total=(2**data[0].shape[1]),
                                leave=False)
        try:
            for model, acc, training_time, cols in results:
                pickled = codecs.encode(pickle.dumps(model, protocol=4),
                                        "base64").decode()
                # The following returns the saved model from a base64 string
                # saved into the variable "pickled".
                # >>> pickle.loads(codecs.decode(pickled.encode(), "base64"))
                res.append({
                    'model': pickled,
                    'columns': cols,
                    'accuracy': {
                        'absolute': f'{acc[0]}/{len(labels[1])}',
                        'relative': acc[1]
                    },
                    'time': training_time  # in seconds
                })
        except KeyboardInterrupt:
            logger.warning("Received SIGINT! Stopping early")

    end_time = time.time()

    return res, end_time - start_time
