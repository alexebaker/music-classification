from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import os
import sys
import sunau
import scipy
import soundfile as sf
import numpy as np

from scikits.talkbox.features import mfcc

from music_classifier import BASE_DIR


genre_dir = os.path.join(BASE_DIR, 'data', 'genres')

total_files = 900
max_audio_len = 675808
min_audio_len = 66000


def read_music_files(folder=genre_dir, use_auread=True):
    """Reads the music files.
    """
    file_count = 0
    audio_data = np.zeros((total_files, min_audio_len), dtype=np.float64)
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith('.au'):
                aufile = os.path.join(root, f)
                data, samplerate = sf.read(aufile)
                audio_data[file_count, :] = data[:min_audio_len]
                file_count += 1
    return audio_data


def get_fft_features(audio_data):
    """Get the FFT features from the data set
    """
    return abs(scipy.fft(audio_data)[:1000])


def get_mfcc_features(audio_data):
    """Get the MFCC features from the data set
    """
    ceps, _, _ = mfcc(audio_data)
    return ceps


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
