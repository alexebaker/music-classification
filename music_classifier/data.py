from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import os
import sys
import scipy
import soundfile as sf
import numpy as np

from scikits.talkbox.features import mfcc

from music_classifier import BASE_DIR


genre_dir = os.path.join(BASE_DIR, 'data', 'genres')
audio_data_file = os.path.join(BASE_DIR, 'data', 'audio_data.npy')

total_files = 900
max_audio_len = 675808
min_audio_len = 66000
fft_feature_range = (0, 1000)


genre_mapping = {
    'blues': 0,
    'classical': 1,
    'country': 2,
    'disco': 3,
    'hiphop': 4,
    'jazz': 5,
    'metal': 6,
    'pop': 7,
    'reggae': 8,
    'rock': 9}


def read_music_files(folder=genre_dir, data_file=audio_data_file):
    """Reads the music files.
    """
    if os.path.exists(data_file):
        audio_data = np.load(data_file)
    else:
        file_count = 0
        audio_data = np.zeros((total_files, min_audio_len+1), dtype=np.float64)
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.endswith('.au'):
                    aufile = os.path.join(root, f)
                    data, samplerate = sf.read(aufile)

                    audio_data[file_count, :min_audio_len] = data[:min_audio_len]

                    genre = os.path.basename(os.path.normpath(root))
                    audio_data[file_count, -1] = _genre_2_id(genre)

                    file_count += 1
        np.save(data_file, audio_data)
    return audio_data


def get_fft_features(audio_data):
    """Get the FFT features from the data set
    """
    start = fft_feature_range[0]
    end = fft_feature_range[1]
    fft_features = np.zeros((audio_data.shape[0], end-start+1), dtype=np.float64)

    fft_features[:, :-1] = np.abs(scipy.fftpack.fft(audio_data[:, :-1], axis=1))[:, start:end]
    fft_features[:, -1] = audio_data[:, -1]
    return fft_features


def get_mfcc_features(audio_data):
    """Get the MFCC features from the data set
    """
    audio_data[audio_data == 0] = 1
    ceps, _, _ = mfcc(audio_data[0, :-1])
    print(ceps)
    print(ceps.shape)
    return ceps


def get_custrom_features():
    """Get custom features from the data set
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


def _genre_2_id(genre):
    return genre_mapping[genre]


def _id_2_genre(genre_id):
    for genre, gid in genre_mapping.iteritems():
        if gid == genre_id:
            return genre
    return "blues"
