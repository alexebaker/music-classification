from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division


import argparse

from music_classifier import data


def parse_args():
    """Parse CLI arguments.

    :rtype: dict
    :returns: Dictonairy of parsed cli arguments.
    """

    # argument parser object
    parser = argparse.ArgumentParser(
        description='Classifies the testing data using naive bayes and the training data.')

    # Add arguments to the parser
    parser.add_argument(
        '--training-data',
        type=str,
        default=data.genre_dir,
        help='Path to the training data folder.')

    parser.add_argument(
        '--validation-data',
        type=str,
        default=data.validation_dir,
        help='Path to the validation data folder.')

    parser.add_argument(
        '--fft',
        action='store_true',
        help='Path to the validation data folder.')

    parser.add_argument(
        '--mfcc',
        action='store_true',
        help='Path to the validation data folder.')

    parser.add_argument(
        '--custom',
        action='store_true',
        help='Path to the validation data folder.')

    parser.add_argument(
        '--lr',
        action='store_true',
        help='Path to the validation data folder.')

    parser.add_argument(
        '--knn',
        action='store_true',
        help='Path to the validation data folder.')

    parser.add_argument(
        '--all',
        action='store_true',
        help='Path to the validation data folder.')

    return vars(parser.parse_args())
