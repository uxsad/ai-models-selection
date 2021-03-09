import itertools
from typing import Dict, Any

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

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = {
    "forest": sk.ensemble.RandomForestClassifier(),
    "svm": sk.svm.SVC(),
    "adaboost": sk.ensemble.AdaBoostClassifier(),
    "tree": sk.tree.DecisionTreeClassifier(),
    "perceptron": sk.neural_network.MLPClassifier(),
}

CV_FOLD = 10


def execute_model(data: pd.DataFrame, labels: pd.Series, algorithm: sk.base.BaseEstimator) -> Dict[str, Any]:
    start_time = time.time()

    output = sk.model_selection.cross_validate(algorithm, data[0], labels[0], cv=CV_FOLD, return_estimator=True)
    logger.debug("Completed cross validation")

    end_time = time.time()

    return output#\
           #accuracy(data[1], labels[1], output["output_model"], algorithm), \
           #end_time - start_time, \
           #list(data[0].columns.values) if hasattr(data[0], 'columns') else data[0].shape[1]


def accuracy(test_data, test_labels, model, algorithm):
    test = algorithm(input_model=model, test=test_data)
    correct = np.sum(test['predictions'] == np.reshape(test_labels, (
        test_labels.shape[0],)))

    return correct, 100 * float(correct) / float(len(test_labels))
