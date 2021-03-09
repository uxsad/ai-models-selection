import itertools
import tqdm
import codecs
import functools
import json
import multiprocessing as mp
import pickle
import time
import logging

import sklearn as sk
import mlxtend as mlx
from mlxtend import feature_selection
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
    start_time = time.time()
    res = {}
    try:
        sfs = mlx.feature_selection.SequentialFeatureSelector(algorithm,
                                                              k_features="best",
                                                              cv=10,
                                                              n_jobs=n_jobs,
                                                              verbose=2, scoring='accuracy')
        out = sfs.fit(data[0], labels[0])
        # pickled = codecs.encode(pickle.dumps(model, protocol=4), "base64").decode()
        # feature_set = [data[0].columns.get_loc(col) for col in cols]
        res = {
            # 'model': pickled,
            'columns': out.k_feature_names_,
            'score': out.k_score_,
            'all': out.subsets_
        }
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
    n_sets = 2 ** data[0].shape[1]
    with mp.Pool(n_jobs, initializer=initializer) as pool:
        g = functools.partial(execute_model,
                              labels=labels,
                              algorithm=algorithm)
        results = pool.imap_unordered(g, columns_set, chunksize=(300))
        if show_progress:
            results = tqdm.tqdm(results,
                                desc="Features",
                                total=n_sets,
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


def pca(data, labels, algorithm, n_jobs=1, show_progress=False):
    def initializer():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    res = []
    labels = (labels[0].to_numpy()[..., None], labels[1].to_numpy()[..., None])
    g = functools.partial(execute_model, labels=labels, algorithm=algorithm)
    if n_jobs == 1:
        logger.info("Starting in serial mode")
    else:
        logger.info("Starting %d parallel jobs", n_jobs)
    start_time = time.time()
    try:
        dimensionalities = tqdm.trange(data[0].shape[1], desc="Dimensions",
                                       leave=False) if show_progress else range(
            data[0].shape[1])
        pca_datasets = []
        for dim in dimensionalities:
            get_pca = sk.decomposition.PCA(n_components=dim + 1)
            get_pca.fit(data[0])
            pca_output = (get_pca.transform(data[0]), get_pca.transform(data[1]))
            pca_datasets.append(pca_output)

        if show_progress:
            pca_datasets = tqdm.tqdm(pca_datasets,
                                     desc="Training",
                                     leave=False)
        if n_jobs == 1:
            results = map(g, pca_datasets)
        else:
            with mp.Pool(n_jobs, initializer=initializer) as pool:
                results = pool.imap_unordered(g, pca_datasets, chunksize=int(len(pca_datasets) / n_jobs))
        if show_progress:
            results = tqdm.tqdm(results, desc="Training", leave=False, total=len(pca_datasets))
        try:
            for model, acc, training_time, n_cols in results:
                pickled = codecs.encode(pickle.dumps(model, protocol=4),
                                        "base64").decode()
                # The following returns the saved model from a base64 string
                # saved into the variable "pickled".
                # >>> pickle.loads(codecs.decode(pickled.encode(), "base64"))
                res.append({
                    'model': pickled,
                    'accuracy': {
                        'absolute': f'{acc[0]}/{len(labels[1])}',
                        'relative': acc[1]
                    },
                    'components': n_cols,
                    'time': training_time  # in seconds
                })
        except KeyboardInterrupt:
            logger.warning("Received SIGINT! Stopping early")
    except KeyboardInterrupt:
        logger.warning("Received SIGINT! Stopping early")
    end_time = time.time()
    return res, end_time - start_time
