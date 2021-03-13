import base64
import logging
import os
import pathlib
import pickle
import time
from typing import Dict, List, Union, Tuple

import click
import coloredlogs
import mlxtend as mlx
import numpy as np
import pandas as pd
import sklearn as sk
import yaml
from mlxtend import feature_selection
from sklearn import ensemble, svm, tree, neural_network

logger = logging.getLogger("ai_models")

AVAILABLE_MODELS = {
    "forest": sk.ensemble.RandomForestClassifier(),
    "svm": sk.svm.SVC(),
    "adaboost": sk.ensemble.AdaBoostClassifier(),
    "tree": sk.tree.DecisionTreeClassifier(),
    "perceptron": sk.neural_network.MLPClassifier(),
}

CV_FOLD = 10

KEYS_TO_INCLUDE = {
    "middle.url", "middle.url.category", "middle.id", 'middle.user_id'
}
KEYS_TO_PREDICT = {
    "middle.emotions.joy", "middle.emotions.fear", "middle.emotions.disgust",
    "middle.emotions.sadness", "middle.emotions.anger",
    "middle.emotions.valence", "middle.emotions.surprise",
    "middle.emotions.contempt", "middle.emotions.engagement"
}
KEYS_TO_IGNORE = {
    "middle.emotions.exists",  # Always True by design
    # The time of the middle object has served its purpose in the
    # creation of
    # the interval and is now useless
    "middle.timestamp",
    # Often contains untreatable values (NaN or Inf) by
    # design
    "middle.trajectory.slope",
}


def load(path: pathlib.Path, emotion: str, width: int = None, location: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the dataset."""

    def can_take_column(col: str) -> bool:
        if col in KEYS_TO_IGNORE:
            return False
        if col in KEYS_TO_INCLUDE | KEYS_TO_PREDICT:
            return True

        if not width and not location:
            return True
        if width and not location:
            return col.startswith(f"{width}.")
        if not width and location:
            return f".{location}." in col

        return col.startswith(f"{width}.{location}.")

    target = f"middle.emotions.{emotion}"
    if target not in KEYS_TO_PREDICT:
        raise ValueError("{emotion} is not a supported emotion")

    users = pd.read_csv(path / 'users.csv',
                        index_col='id',
                        dtype={
                            'age': np.float32,
                            'internet': np.float32,
                            'gender': str,
                        })
    users['gender'] = pd.Categorical(users['gender'],
                                     categories={'m', 'f', 'a'})
    users['age'] = pd.Categorical(users['age'], categories=range(6))

    websites = pd.read_csv(path / 'websites.csv',
                           dtype={
                               'count': np.float32,
                               'category': str
                           })
    websites['category'] = pd.Categorical(websites['category'])
    websites['url'] = pd.Categorical(websites['url'])

    df = pd.read_csv(
        path / f'{emotion}.csv',
        engine='c',
        usecols=can_take_column,
        encoding='utf-8',
        index_col="middle.id",
    )
    df['user.age'] = df['middle.user_id'].map(users['age'])
    df['user.internet'] = df['middle.user_id'].map(users['internet'])
    df['user.gender'] = df['middle.user_id'].map(users['gender'])
    # The user id is no longer needed
    df.drop(columns=['middle.user_id'], inplace=True)

    # OneHot encoder
    df['user.gender'] = df['user.gender'].map({
        'm': 'male',
        'f': 'female',
        'a': 'other'
    })
    df = pd.get_dummies(df, prefix_sep='.', columns=['user.gender'])
    # Ordinal encoder
    df['middle.url'] = df['middle.url'].map(
        {u: i
         for i, u in enumerate(pd.Categorical(df['middle.url']))})
    # Ordinal encoder
    df['middle.url.category'] = df['middle.url.category'].map({
        c: i
        for i, c in enumerate(pd.Categorical(df['middle.url.category']))
    })

    # Filling NaN values on target columns: by design if a value isn't there it
    # means that it was under 1 and can then be approximated to 0 (see
    # https://shorturl.at/JOTU5 and https://shorturl.at/etM09).
    x = df.drop(columns=KEYS_TO_PREDICT)
    y = df[KEYS_TO_PREDICT].fillna(0)

    return x, y[target]


def test_report(data: Tuple[pd.DataFrame, pd.DataFrame], labels: Tuple[pd.Series, pd.Series],
                algorithm: sk.base.ClassifierMixin) \
        -> Dict[str, Union[float, Dict[str, List[float]]]]:
    test_algorithm_copy = sk.base.clone(algorithm)
    start_time = time.time()
    test_algorithm_copy.fit(data[0], labels[0])
    output = sk.metrics.classification_report(labels[1], test_algorithm_copy.predict(data[1]), output_dict=True)
    logger.debug("Completed final test")
    final_test = time.time() - start_time
    output["time"] = final_test
    pickled = base64.b64encode(pickle.dumps(test_algorithm_copy, protocol=4)).decode()
    output['model'] = pickled

    return output


def forward_feature_selection(data, labels, algorithm, n_jobs=1, show_progress=False):
    start_time = time.time()
    sfs = mlx.feature_selection.SequentialFeatureSelector(algorithm,
                                                          k_features="best",
                                                          cv=10,
                                                          n_jobs=n_jobs,
                                                          verbose=2 if show_progress else 0,
                                                          scoring='accuracy')
    out = sfs.fit(data[0], labels[0])
    logger.debug("Completed SFFS")

    output = test_report(data, labels, algorithm)

    res = {
        'final': output,
        'columns': list(out.k_feature_names_),
        'score': float(out.k_score_),
        'all': out.subsets_
    }
    # Convert all numpy's types to Python's ones for better compatibility with other libraries (ie. YAML)
    for key in res['all']:
        res['all'][key]['cv_scores'] = res['all'][key]['cv_scores'].tolist()
        # res['all'][key]['feature_idx'] = list(res['all'][key]['feature_idx'])
        del res['all'][key]['feature_idx']
        res['all'][key]['feature_names'] = list(res['all'][key]['feature_names'])
        res['all'][key]['avg_score'] = float(res['all'][key]['avg_score'])
    end_time = time.time()
    return res, end_time - start_time


def exhaustive_feature_selection(data, labels, algorithm, n_jobs=1):
    start_time = time.time()
    sfs = mlx.feature_selection.ExhaustiveFeatureSelector(algorithm,
                                                          cv=10,
                                                          n_jobs=n_jobs,
                                                          scoring='accuracy')
    out = sfs.fit(data[0], labels[0])
    logger.debug("Completed EFS")

    output = test_report(data, labels, algorithm)

    res = {
        'final': output,
        'columns': out.best_feature_names_,
        'score': out.best_score_,
        'all': out.subsets_
    }
    end_time = time.time()
    return res, end_time - start_time


def pca(data, labels, algorithm):
    # Uses Minka's MLE - Thomas P. Minka, "Automatic choice of dimensionality for PCA", NIPS 2000
    process_start_time =time.time()
    get_pca = sk.decomposition.PCA(n_components='mle')
    get_pca.fit(data[0])
    pca_output = (get_pca.transform(data[0]), get_pca.transform(data[1]))

    res = {}

    cv_algorithm_copy = sk.base.clone(algorithm)
    start_time = time.time()
    res["cross_val"] = sk.model_selection.cross_validate(cv_algorithm_copy, pca_output[0], labels[0], cv=CV_FOLD)
    logger.debug("Completed cross validation")
    cv_time = time.time() - start_time
    # Convert all numpy's arras to Python's lists for better compatibility with other libraries (ie. YAML)
    for key in res["cross_val"]:
        res["cross_val"][key] = res["cross_val"][key].tolist()
    res["cross_val"]["time"] = cv_time

    res['final'] = test_report(pca_output, labels, algorithm)
    res["n_components"] = int(get_pca.n_components_)
    return res, time.time() - process_start_time


@click.command()
@click.argument('dataset', type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument('emotion',
                type=click.Choice([e.split('.')[2] for e in KEYS_TO_PREDICT],
                                  case_sensitive=False))
@click.argument('width', type=int)
@click.argument('location', type=click.Choice(['before', 'after', 'full'],
                                              case_sensitive=False))
@click.option('--random', '-r', type=int, default=0, help='Set the random seed')
@click.option('--out', '-o', type=click.File(mode='w'), default='-',
              help="Set the path where the model will be saved. A new file with the"
                   " same name as the model's emotion will be created inside this"
                   " directory.")
@click.option('--verbose', '-V', help='Run verbosely', is_flag=True)
@click.option('--progress', '-P', help='Show progress', is_flag=True)
@click.option('--jobs', '-j', type=int, default=1,
              help='Number of parallel jobs')
@click.option('--algorithm', '-a',
              type=click.Choice(AVAILABLE_MODELS.keys(), case_sensitive=False),
              default=[list(AVAILABLE_MODELS.keys())[0]], help='The algorithm')
@click.option('--strategy', '-s',
              type=click.Choice(['sfs', 'pca', 'efs'], case_sensitive=False),
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
    data, labels = load(pathlib.Path(dataset), emotion, width=width, location=location)
    logger.info("Loaded a dataset of size %s", data.shape)

    if os.getenv('UXSAD_ENV') == 'test':
        logger.warning("Started in test mode")
        data = data.iloc[0:300, :]
        labels = labels.iloc[0:300]

    logger.info("Splitting the dataset into train and test set (test ratio: %.2f%%)", 100 * 0.3)
    train_data, test_data, train_labels, test_labels = sk.model_selection.train_test_split(data, labels, train_size=0.7,
                                                                                           random_state=random)
    logger.info("Done. Train size: %s, Test size: %s", train_data.shape, test_data.shape)

    res = {}
    if "sfs" in strategy:
        logger.info("Starting forward feature selection")
        res['feature_selection'], duration = forward_feature_selection(
            (train_data, test_data), (train_labels, test_labels),
            AVAILABLE_MODELS[algorithm],
            show_progress=progress,
            n_jobs=jobs)
        logger.info("Forward feature selection completed. Took %.2f seconds",
                    duration)
    if "efs" in strategy:
        logger.info("Starting exhaustive feature selection")
        res['exhaustive_feature_selection'], duration = exhaustive_feature_selection(
            (train_data, test_data), (train_labels, test_labels),
            AVAILABLE_MODELS[algorithm],
            n_jobs=jobs)
        logger.info("Forward feature selection completed. Took %.2f seconds",
                    duration)
    if "pca" in strategy:
        logger.info("Starting PCA")
        res['pca'], duration = pca(
            (train_data, test_data), (train_labels, test_labels),
            AVAILABLE_MODELS[algorithm])
        logger.info("PCA completed. Took %.2f seconds", duration)

    out.write(yaml.dump(res))


if __name__ == '__main__':
    main()
