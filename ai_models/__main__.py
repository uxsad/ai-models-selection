import pickle
import coloredlogs
import logging
import mlpack
import ai_models.cli as cli
import ai_models.dataset as dataset
import ai_models.model as model
import ai_models.model.preprocess
import numpy as np
import os
import yaml

logger = logging.getLogger("ai_models")


def main(*args):
    """The main entry point of the tool."""
    args = cli.cli(args)

    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s (%(name)s) %(message)s", )
    if args.verbose:
        logger.setLevel(logging.INFO)
        coloredlogs.install(
            level=logging.INFO,
            logger=logger,
            fmt="[%(levelname)s] %(asctime)s (%(name)s) %(message)s",
        )

    logger.info("Loading the dataset from '%s'", args.dataset)
    data, labels = dataset.load(args.dataset,
                                "engagement",
                                width=args.width,
                                location=args.location)
    logger.info("Loaded a dataset of size %s", data.shape)

    if os.getenv('UXSAD_ENV') == 'test':
        logger.warning("Started in test mode")
        data = data.head()

    logger.info(
        "Splitting the dataset into train and test set (test ratio: %.2f%%)",
        100 * 0.3)
    train_data, train_labels, test_data, test_labels = \
        model.preprocess.split_dataset(data, labels)
    logger.info("Done. Train size: %s, Test size: %s", train_data.shape,
                test_data.shape)

    res = {}
    logger.info("Starting exhaustive feature selection")
    res['exhaustive_feature_selection'], duration = model.feature_selection(
        (train_data, test_data), (train_labels, test_labels),
        model.AVAILABLE_MODELS[args.model],
        show_progress=args.verbose,
        random=args.random,
        n_jobs=args.jobs)
    logger.info("Exhaustive feature selection completed. Took %.2f seconds",
                duration)
    print(yaml.dump(res))


if __name__ == '__main__':
    main()
