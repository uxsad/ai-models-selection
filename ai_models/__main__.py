import pickle
import coloredlogs
import logging
import ai_models.dataset as dataset
import ai_models.model as model
import numpy as np
import os
import yaml
import click
import pathlib
import itertools
from typing import Dict, Any, List, Union, Tuple

import tqdm
import codecs
import functools
import json
import multiprocessing as mp
import pickle
import time
import logging

import sklearn as sk
from sklearn import ensemble, neural_network, svm, tree
import numpy as np
import yaml
import pandas as pd
import signal
from ai_models.dataset import KEYS_TO_PREDICT

logger = logging.getLogger("ai_models")

AVAILABLE_MODELS = {
    "forest": sk.ensemble.RandomForestClassifier(),
    "svm": sk.svm.SVC(),
    "adaboost": sk.ensemble.AdaBoostClassifier(),
    "tree": sk.tree.DecisionTreeClassifier(),
    "perceptron": sk.neural_network.MLPClassifier(),
}

CV_FOLD = 10


def execute_model(data: Tuple[pd.DataFrame, pd.DataFrame], labels: Tuple[pd.Series, pd.Series],
                  algorithm: sk.base.ClassifierMixin) \
        -> Dict[str, Union[float, Dict[str, List[float]]]]:
    output = {}
    cv_algorithm_copy = sk.base.clone(algorithm)
    start_time = time.time()
    output["cross_val"] = sk.model_selection.cross_validate(cv_algorithm_copy, data[0], labels[0], cv=CV_FOLD)
    logger.debug("Completed cross validation")
    cv_time = time.time() - start_time
    # Convert all numpy's arras to Python's lists for better compatibility with other libraries (ie. YAML)
    for key in output["cross_val"]:
        output[key] = output[key].tolist()
    output["cross_val"]["time"] = cv_time

    test_algorithm_copy = sk.base.clone(algorithm)
    start_time = time.time()
    test_algorithm_copy.fit(data[0], labels[0])
    output["final"] = sk.metrics.classification_report(labels[1], test_algorithm_copy.predict(data[1]), output_dict=True)
    logger.debug("Completed final test")
    final_test = time.time() - start_time
    output["final"]["time"] = final_test

    return output


@click.command()
@click.argument('dataset',
                type=click.Path(exists=True, dir_okay=True, file_okay=False,
                                path_type=pathlib.Path))
@click.argument('emotion',
                type=click.Choice([e.split('.')[2] for e in KEYS_TO_PREDICT],
                                  case_sensitive=False))
@click.argument('width', type=int)
@click.argument('location', type=click.Choice(['before', 'after', 'full'],
                                              case_sensitive=False))
@click.option('--random', '-r', type=int, default=0, help='Set the random seed')
@click.option('--out', '-o', type=click.Path(dir_okay=False, file_okay=True,
                                             path_type=pathlib.Path),
              default=None,
              help="Set the path where the model will be saved. A new file with the"
                   " same name as the model's emotion will be created inside this"
                   " directory.")
@click.option('--verbose', '-v', help='Run verbosely', is_flag=True)
@click.option('--progress', '-P', help='Show progress', is_flag=True)
@click.option('--jobs', '-j', type=int, default=1,
              help='Number of parallel jobs')
@click.option('--algorithm', '-a',
              type=click.Choice(AVAILABLE_MODELS.keys(), case_sensitive=False),
              default=[list(AVAILABLE_MODELS.keys())[0]], help='The algorithm',
              multiple=True)
@click.option('--strategy', '-s',
              type=click.Choice(['sfs', 'pca'], case_sensitive=False),
              default=['sfs'], help='The strategy to apply', multiple=True)
def main(dataset, emotion, width, location, random, out, verbose, progress,
         jobs, algorithm, strategy):
    """The main entry point of the tool."""
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s (%(name)s) %(message)s", )
    if verbose:
        logger.setLevel(logging.INFO)
        coloredlogs.install(
            level=logging.INFO,
            logger=logger,
            fmt="[%(levelname)s] %(asctime)s (%(name)s) %(message)s",
        )

    logger.info("Loading the dataset from '%s'", dataset)
    data, labels = dataset.load(dataset,
                                emotion,
                                width=width,
                                location=location)
    logger.info("Loaded a dataset of size %s", data.shape)

    if os.getenv('UXSAD_ENV') == 'test':
        logger.warning("Started in test mode")
        data = data.iloc[0:300, :]

    logger.info("Splitting the dataset into train and test set (test ratio: %.2f%%)", 100 * 0.3)
    train_data, test_data, train_labels, test_labels = sk.model_selection.train_test_split(data, labels, train_size=0.7, random_state=random)
    logger.info("Done. Train size: %s, Test size: %s", train_data.shape, test_data.shape)

    res = {}
    if "sfs" in strategy:
        logger.info("Starting forward feature selection")
        res['feature_selection'], duration = model.forward_feature_selection(
            (train_data, test_data), (train_labels, test_labels),
            AVAILABLE_MODELS[algorithm],
            show_progress=progress,
            n_jobs=jobs)
        logger.info("Forward feature selection completed. Took %.2f seconds",
                    duration)
    if "pca" in strategy:
        logger.info("Starting PCA")
        res['pca'], duration = model.pca(
            (train_data, test_data), (train_labels, test_labels),
            AVAILABLE_MODELS[algorithm],
            show_progress=progress,
            n_jobs=jobs)
        logger.info("PCA completed. Took %.2f seconds", duration)
    print(yaml.dump(res))


if __name__ == '__main__':
    main()
