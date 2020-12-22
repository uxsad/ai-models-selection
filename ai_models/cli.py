import argparse
import pathlib
from ai_models.model import AVAILABLE_MODELS
from ai_models.dataset import KEYS_TO_PREDICT


def cli(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--random",
                        "-r",
                        type=int,
                        default=0,
                        help="Set the random seed")

    parser.add_argument("dataset", help="The dataset", type=pathlib.Path)

    parser.add_argument(
        '--out',
        '-o',
        help="Set the path where the model will be saved. A new file with the"
        " same name as the model's emotion will be created inside this"
        " directory.",
        type=pathlib.Path,
        default=None)

    parser.add_argument('--verbose',
                        '-V',
                        help="Run verbosely",
                        action='store_true')

    parser.add_argument('--progress',
                        '-P',
                        help="Show progress",
                        action='store_true')

    parser.add_argument('--jobs',
                        '-j',
                        help="Number of parallel jobs",
                        type=int,
                        default=1)

    parser.add_argument('emotion',
                        help="The emotion",
                        type=str,
                        choices=[e.split('.')[2] for e in KEYS_TO_PREDICT])
    parser.add_argument('width',
                        help="The width",
                        type=int)
    parser.add_argument('location',
                        help="The location",
                        type=str,
                        choices=["before", "after", "full"])

    parser.add_argument('--model',
                        '-m',
                        help="The model",
                        type=str,
                        default=list(AVAILABLE_MODELS.keys())[0],
                        choices=AVAILABLE_MODELS.keys())

    return parser.parse_args()
