import argparse
import pathlib


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

    parser.add_argument('--jobs',
                        '-j',
                        help="Number of parallel jobs",
                        type=int,
                        default=1)

    return parser.parse_args()
