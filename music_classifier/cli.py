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
        '--features',
        type=str,
        nargs='+',
        help='Feature extraction method to use. Can be multiple features at once. Features can be: fft, mfcc, or dwt.')

    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        help='Classification methods to use. Can be multiple methods at once. Methods can be: lr, knn, svm, or nn.')

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all features and classifiers.')

    parser.add_argument(
        '--cross-validate',
        action='store_true',
        help='Perform 10 fold cross validation.')

    parser.add_argument(
        '--confusion-matrix',
        action='store_true',
        help='Calculate the confusion matrix for the given methods.')

    return vars(parser.parse_args())
