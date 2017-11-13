from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import os
import scipy
import json
import soundfile as sf
import numpy as np
import pywt

from scikits.talkbox.features import mfcc

from sklearn.preprocessing import Normalizer

from music_classifier import BASE_DIR


genre_dir = os.path.join(BASE_DIR, 'data', 'genres')
validation_dir = os.path.join(BASE_DIR, 'data', 'rename')
audio_data_file = os.path.join(BASE_DIR, 'data', 'audio_data.npy')
validation_data_file = os.path.join(BASE_DIR, 'data', 'validation_data.npy')
mapping_file = os.path.join(BASE_DIR, 'data', 'validation_mapping.json')
label_file = os.path.join(BASE_DIR, 'data', 'audio_labels.npy')

total_files = 900
validation_files = 100
min_validation_len = 661504
max_validation_len = 661794
max_audio_len = 675808
min_audio_len = 660000
feature_range = (int(min_audio_len / 10), int(min_audio_len * 9 / 10), 20)


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


def read_music_files(folder=genre_dir, data_file=audio_data_file, label_file=label_file):
    """Reads the music files.
    """
    if os.path.exists(data_file) and os.path.exists(label_file):
        audio_data = np.load(data_file)
        labels = np.load(label_file)
    else:
        file_count = 0
        audio_data = np.zeros((total_files, max_audio_len+1), dtype=np.float64)
        labels = np.zeros((total_files,), dtype=np.int16)
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.endswith('.au'):
                    aufile = os.path.join(root, f)
                    data, samplerate = sf.read(aufile)
                    data_len = len(data) + 1

                    audio_data[file_count, 0] = data_len
                    audio_data[file_count, 1:data_len] = data

                    genre = os.path.basename(os.path.normpath(root))
                    labels[file_count] = _genre_2_id(genre)

                    file_count += 1
        np.save(data_file, audio_data)
        np.save(label_file, labels)
    return audio_data, labels


def get_labels(label_file=label_file):
    if os.path.exists(label_file):
        labels = np.load(label_file)
    else:
        _, labels = read_music_files()
    return labels


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


def get_mapping(mapping_file=mapping_file):
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            validation_mapping = json.load(f)
    else:
        _, validation_mapping = read_validation_files()
    return validation_mapping


def get_features(feature, audio_data, npy_file):
    features = None
    if feature == 'fft':
        features = get_fft_features(audio_data, npy_file)
    elif feature == 'mfcc':
        features = get_mfcc_features(audio_data, npy_file)
    elif feature == 'dwt':
        features = get_dwt_features(audio_data, npy_file)
    return features


def get_fft_features(audio_data, npy_file):
    """Get the FFT features from the data set
    """
    start = feature_range[0] + 1
    end = feature_range[1] + 1
    step = feature_range[2]

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
            data_len = int(audio_data[row, 0])
            ceps, _, _ = mfcc(audio_data[row, 1:data_len])
            num_ceps = ceps.shape[0]
            ceps_features[row, :] = np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0)
        np.save(npy_file, ceps_features)
    return ceps_features


def get_dwt_features(audio_data, npy_file):
    """Get dwt features from the data set
    """
    audio_data[audio_data == 0] = 1
    start = feature_range[0] + 1
    end = feature_range[1] + 1
    step = feature_range[2]

    if os.path.exists(npy_file):
        dwt_features = np.load(npy_file)
    else:
        dwt_features = np.zeros((audio_data.shape[0], max_audio_len+1), dtype=np.float64)
        for row in range(audio_data.shape[0]):
            data_len = int(audio_data[row, 0])
            cA, cD = pywt.dwt(audio_data[row, 1:data_len], 'db2')
            dwt_features[row, 1:data_len] = pywt.idwt(cA, cD, 'db2')
            dwt_features[row, 0] = data_len
        np.save(npy_file, dwt_features)
    return dwt_features[:, start:end:step]


def normalize_data(data):
    norm = Normalizer()
    return norm.fit_transform(data)


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
