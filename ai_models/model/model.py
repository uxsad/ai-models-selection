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


def execute_model(data, labels, model, random=0):
    start_time = time.time()

    output = model(labels=labels[0], training=data[0], seed=random)
    logger.debug("Completed training")

    end_time = time.time()

    return output["output_model"], \
            accuracy(data[1], labels[1], output["output_model"], model), \
            end_time - start_time, \
            list(data[0].columns.values)


def accuracy(test_data, test_labels, model, f):
    test = f(input_model=model, test=test_data)
    correct = np.sum(test['predictions'] == np.reshape(test_labels, (
        test_labels.shape[0], )))

    return correct, 100 * float(correct) / float(len(test_labels))


def feature_selection(data,
                      labels,
                      model,
                      n_jobs=1,
                      random=0,
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
                              model=model,
                              random=random)
        results = pool.imap_unordered(g, columns_set, chunksize=(300))

        if show_progress:
            results = tqdm.tqdm(results,
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
