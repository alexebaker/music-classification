from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np

from music_classifier import cli
from music_classifier import data
from music_classifier import classifiers
from music_classifier import BASE_DIR


fft_file = os.path.join(BASE_DIR, 'data', 'fft_features.npy')
mfcc_file = os.path.join(BASE_DIR, 'data', 'mfcc_features.npy')
dwt_file = os.path.join(BASE_DIR, 'data', 'dwt_features.npy')
fft_validation_file = os.path.join(BASE_DIR, 'data', 'fft_validation_features.npy')
mfcc_validation_file = os.path.join(BASE_DIR, 'data', 'mfcc_validation_features.npy')
dwt_validation_file = os.path.join(BASE_DIR, 'data', 'dwt_validation_features.npy')

data_dir = os.path.join(BASE_DIR, 'data')


def main():
    """Main entry point."""
    # Parse the command line arguments
    cli_args = cli.parse_args()

    audio_data = None
    validation_data = None
    labels = data.get_labels()
    validation_mapping = data.get_mapping()

    runall = cli_args['all']
    if runall:
        features = ['fft', 'mfcc', 'dwt']
        methods = ['lr', 'knn', 'svm', 'nn']
    else:
        features = cli_args['features']
        methods = cli_args['methods']

    for feature in features:
        feature_file = os.path.join(data_dir, "%s_features.npy" % feature)
        feature_validation_file = os.path.join(data_dir, "%s_validation_features.npy" % feature)

        if not os.path.exists(feature_file):
            if audio_data is None:
                audio_data, labels = data.read_music_files(folder=cli_args['training_data'])

        if not os.path.exists(feature_validation_file):
            if validation_data is None:
                validation_data, validaion_mapping = data.read_validation_files(cli_args['validation_data'])

        feature_data = data.normalize_data(data.get_features(feature, audio_data, feature_file))
        feature_validation_data = data.normalize_data(data.get_features(feature, validation_data, feature_validation_file))

        for method in methods:
            classifier = classifiers.train_data(method, feature_data, labels)
            classification = classifiers.classify_data(classifier, feature_validation_data)
            classification_file = "%s_%s_classification.csv" % (feature, method)
            data.save_classification(classification, classification_file, validation_mapping)
    return


if __name__ == "__main__":
    main()
