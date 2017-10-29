from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import os


def read_music_files():
    """Reads the music files.
    """
    return


def get_fft_features():
    """Get the FFT features from the data set
    """
    return


def get_mfcc_features():
    """Get the MFCC features from the data set
    """
    return


def get_custrom_features():
    """Get the MFCC features from the data set
    """
    return


def save_classification(classification, classification_file):
    """Saves the classification from the classification algorithm.

    :type classification: list
    :param classification: The classification output from the classifier.

    :type classification_file: File Object
    :param classification_file: File to write the classification to.
    """
    print("id,class", file=classification_file)
    for row in classification:
        print("%d,%d" % (row[0], row[1]), file=classification_file)
    return
