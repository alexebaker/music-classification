from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import os
import sys
import scipy
import json
import soundfile as sf
import numpy as np

from scikits.talkbox.features import mfcc

from music_classifier import BASE_DIR


genre_dir = os.path.join(BASE_DIR, 'data', 'genres')
validation_dir = os.path.join(BASE_DIR, 'data', 'rename')
audio_data_file = os.path.join(BASE_DIR, 'data', 'audio_data.npy')
validation_data_file = os.path.join(BASE_DIR, 'data', 'validation_data.npy')
mapping_file = os.path.join(BASE_DIR, 'data', 'validation_mapping.json')

total_files = 900
validation_files = 100
min_validation_len = 661504
max_validation_len = 661794
max_audio_len = 675808
min_audio_len = 660000
fft_feature_range = (int(min_audio_len / 10), int(min_audio_len * 9 / 10), 20)


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
        audio_data = np.zeros((total_files, max_audio_len+2), dtype=np.float64)
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.endswith('.au'):
                    aufile = os.path.join(root, f)
                    data, samplerate = sf.read(aufile)
                    data_len = len(data) + 1

                    audio_data[file_count, 0] = data_len
                    audio_data[file_count, 1:data_len] = data

                    genre = os.path.basename(os.path.normpath(root))
                    audio_data[file_count, -1] = _genre_2_id(genre)

                    file_count += 1
        np.save(data_file, audio_data)
    return audio_data[:, :-1], audio_data[:, -1]


def read_validation_files(folder=validation_dir, data_file=validation_data_file, mapping_file=mapping_file):
    """Reads the validation music files.
    """
    if os.path.exists(data_file) and os.path.exists(mapping_file):
        validation_data = np.load(data_file)
        with open(mapping_file, 'r') as f:
            validation_mapping = json.load(f)
    else:
        file_count = 0
        validation_mapping = {}
        validation_data = np.zeros((validation_files, max_validation_len+1), dtype=np.float64)
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.endswith('.au'):
                    aufile = os.path.join(root, f)
                    data, samplerate = sf.read(aufile)
                    data_len = len(data) + 1

                    validation_data[file_count, 0] = data_len
                    validation_data[file_count, 1:data_len] = data
                    validation_mapping[str(file_count)] = f

                    file_count += 1
        np.save(data_file, validation_data)
        with open(mapping_file, 'w') as f:
            json.dump(validation_mapping, f)
    return validation_data, validation_mapping


def get_fft_features(audio_data, npy_file, fft_feature_range=fft_feature_range):
    """Get the FFT features from the data set
    """
    start = fft_feature_range[0] + 1
    end = fft_feature_range[1] + 1
    step = fft_feature_range[2]

    if os.path.exists(npy_file):
        fft_features = np.load(npy_file)
    else:
        fft_features = np.zeros((audio_data.shape[0], max_audio_len+1), dtype=np.float64)

        for row in range(audio_data.shape[0]):
            data_len = int(audio_data[row, 0])
            fft_features[row, 1:data_len] = np.abs(scipy.fftpack.fft(audio_data[row, 1:data_len]))
            fft_features[row, 0] = data_len
        np.save(npy_file, fft_features)
    return fft_features[:, start:end:step]


def get_mfcc_features(audio_data, npy_file):
    """Get the MFCC features from the data set
    """
    audio_data[audio_data == 0] = 1

    if os.path.exists(npy_file):
        ceps_features = np.load(npy_file)
    else:
        ceps_features = np.zeros((audio_data.shape[0], 13), dtype=np.float64)

        for row in range(audio_data.shape[0]):
            #ceps, _, _ = mfcc(audio_data[row, :], fs=22050)
            data_len = int(audio_data[row, 0])
            ceps, _, _ = mfcc(audio_data[row, 1:data_len])
            num_ceps = ceps.shape[0]
            ceps_features[row, :] = np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10), :], axis=0)
        np.save(npy_file, ceps_features)
    return ceps_features


def get_custom_features():
    """Get custom features from the data set
    """
    return


def save_classification(classification, classification_file, validation_mapping):
    """Saves the classification from the classification algorithm.

    :type classification: list
    :param classification: The classification output from the classifier.

    :type classification_file: File Object
    :param classification_file: File to write the classification to.
    """
    with open(classification_file, 'w') as f:
        print("id,class", file=f)
        for idx, label in enumerate(classification):
            print("%s,%s" % (validation_mapping[str(idx)], _id_2_genre(label)),
                  file=f)
    return


def _genre_2_id(genre):
    return genre_mapping[genre]


def _id_2_genre(genre_id):
    for genre, gid in genre_mapping.iteritems():
        if gid == genre_id:
            return genre
    return "blues"
