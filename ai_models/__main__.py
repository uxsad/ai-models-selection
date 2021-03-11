import pickle
import coloredlogs
import logging
import ai_models.dataset as dataset
import ai_models.model as model
import ai_models.model.preprocess
import numpy as np
import os
import yaml
import click
import pathlib
from ai_models.model import AVAILABLE_MODELS
from ai_models.dataset import KEYS_TO_PREDICT

logger = logging.getLogger("ai_models")


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

    logger.info(
        "Splitting the dataset into train and test set (test ratio: %.2f%%)",
        100 * 0.3)
    train_data, train_labels, test_data, test_labels = model.preprocess.split_dataset(
        data, labels, frac=0.7)
    logger.info("Done. Train size: %s, Test size: %s", train_data.shape,
                test_data.shape)

    res = {}
    if "sfs" in strategy:
        logger.info("Starting forward feature selection")
        res['feature_selection'], duration = model.forward_feature_selection(
            (train_data, test_data), (train_labels, test_labels),
            model.AVAILABLE_MODELS[algorithm],
            show_progress=progress,
            n_jobs=jobs)
        logger.info("Forward feature selection completed. Took %.2f seconds",
                    duration)
    if "pca" in strategy:
        logger.info("Starting PCA")
        res['pca'], duration = model.pca(
            (train_data, test_data), (train_labels, test_labels),
            model.AVAILABLE_MODELS[algorithm],
            show_progress=progress,
            n_jobs=jobs)
        logger.info("PCA completed. Took %.2f seconds", duration)
    print(yaml.dump(res))


if __name__ == '__main__':
    main()
