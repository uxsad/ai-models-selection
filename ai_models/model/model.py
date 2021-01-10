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

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = {
    "forest": mlpack.random_forest,
    "svm": mlpack.linear_svm,
    "adaboost":mlpack.adaboost,
    "tree": mlpack.decision_tree,
    "perceptron": mlpack.perceptron,
}


def execute_model(data, labels, algorithm):
    start_time = time.time()

    output = algorithm(labels=labels[0], training=data[0])
    logger.debug("Completed training")

    end_time = time.time()

    return output["output_model"], \
            accuracy(data[1], labels[1], output["output_model"], algorithm), \
            end_time - start_time, \
            list(data[0].columns.values) if hasattr(data[0], 'columns') else data[0].shape[1]


def accuracy(test_data, test_labels, model, algorithm):
    test = algorithm(input_model=model, test=test_data)
    correct = np.sum(test['predictions'] == np.reshape(test_labels, (
        test_labels.shape[0], )))

    return correct, 100 * float(correct) / float(len(test_labels))
