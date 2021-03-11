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
            'columns': list(out.k_feature_names_),
            'score': float(out.k_score_),
            'all': out.subsets_
        }
        # Convert all numpy's types to Python's ones for better compatibility with other libraries (ie. YAML)
        for key in res['all']:
            res['all'][key]['cv_scores'] = res['all'][key]['cv_scores'].tolist()
            #res['all'][key]['feature_idx'] = list(res['all'][key]['feature_idx'])
            del res['all'][key]['feature_idx']
            res['all'][key]['feature_names'] = list(res['all'][key]['feature_names'])
            res['all'][key]['avg_score'] = float(res['all'][key]['avg_score'])
    except KeyboardInterrupt:
        logger.warning("Received SIGINT! Stopping early")
    end_time = time.time()
    return res, end_time - start_time


def exhaustive_feature_selection(data,
                                 labels,
                                 algorithm,
                                 n_jobs=1,
                                 show_progress=False):
    start_time = time.time()
    res = {}
    try:
        sfs = mlx.feature_selection.ExhaustiveFeatureSelector(algorithm,
                                                              cv=10,
                                                              n_jobs=n_jobs,
                                                              scoring='accuracy')
        out = sfs.fit(data[0], labels[0])
        # pickled = codecs.encode(pickle.dumps(model, protocol=4), "base64").decode()
        # feature_set = [data[0].columns.get_loc(col) for col in cols]
        res = {
            # 'model': pickled,
            'columns': out.best_feature_names_,
            'score': out.best_score_,
            'all': out.subsets_
        }
    except KeyboardInterrupt:
        logger.warning("Received SIGINT! Stopping early")
    end_time = time.time()
    return res, end_time - start_time


def pca(data, labels, algorithm, n_jobs=1, show_progress=False):
    # Uses Minka's MLE - Thomas P. Minka, "Automatic choice of dimensionality for PCA", NIPS 2000
    get_pca = sk.decomposition.PCA(n_components='mle')
    get_pca.fit(data[0])
    pca_output = (get_pca.transform(data[0]), get_pca.transform(data[1]))
    start_time = time.time()
    res = execute_model(pca_output, labels, algorithm)
    res["n_components"] = int(get_pca.n_components_)
    end_time = time.time()
    return res, end_time - start_time
