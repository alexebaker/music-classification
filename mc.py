from __future__ import print_function
from __future__ import unicode_literals

import os

from music_classifier import cli
from music_classifier import data
from music_classifier import classifiers
from music_classifier import BASE_DIR


fft_file = os.path.join(BASE_DIR, 'data', 'fft_features.npy')
mfcc_file = os.path.join(BASE_DIR, 'data', 'mfcc_features.npy')
fft_validation_file = os.path.join(BASE_DIR, 'data', 'fft_validation_features.npy')
mfcc_validation_file = os.path.join(BASE_DIR, 'data', 'mfcc_validation_features.npy')


def main():
    """Main entry point."""
    # Parse the command line arguments
    cli_args = cli.parse_args()

    audio_data, labels = data.read_music_files()
    #audio_data = classifiers.data_normalize(audio_data)

    validation_data, validation_mapping = data.read_validation_files()
    #validation_data = classifiers.data_normalize(validation_data)

    runall = cli_args['all']
    if runall or cli_args['fft']:
        features = data.get_fft_features(audio_data, fft_file)
        validation_features = data.get_fft_features(validation_data, fft_validation_file)

        if runall or cli_args['lr']:
            classifier = classifiers.log_reg_train(features, labels)
            classification = classifiers.log_reg_classify(
                classifier,
                validation_features)
            data.save_classification(
                classification,
                "./fft_lr_classification.csv",
                validation_mapping)

        if runall or cli_args['knn']:
            classifier = classifiers.knn_train(features, labels)
            classification = classifiers.knn_classify(
                classifier,
                validation_features)
            data.save_classification(
                classification,
                "./fft_knn_classification.csv",
                validation_mapping)

        if runall or cli_args['svm']:
            classifier = classifiers.svm_train(features, labels)
            classification = classifiers.svm_classify(
                classifier,
                validation_features)
            data.save_classification(
                classification,
                "./fft_svm_classification.csv",
                validation_mapping)

    if runall or cli_args['mfcc']:
        features = data.get_mfcc_features(audio_data, mfcc_file)
        validation_features = data.get_mfcc_features(validation_data, mfcc_validation_file)

        if runall or cli_args['lr']:
            classifier = classifiers.log_reg_train(features, labels)
            classification = classifiers.log_reg_classify(
                classifier,
                validation_features)
            data.save_classification(
                classification,
                "./mfcc_lr_classification.csv",
                validation_mapping)

        if runall or cli_args['knn']:
            classifier = classifiers.knn_train(features, labels)
            classification = classifiers.knn_classify(
                classifier,
                validation_features)
            data.save_classification(
                classification,
                "./mfcc_knn_classification.csv",
                validation_mapping)

        if runall or cli_args['svm']:
            classifier = classifiers.svm_train(features, labels)
            classification = classifiers.svm_classify(
                classifier,
                validation_features)
            data.save_classification(
                classification,
                "./mfcc_svm_classification.csv",
                validation_mapping)

    if False and runall or cli_args['custom']:
        features = data.get_custom_features(audio_data)
        validation_features = data.get_custom_features(validation_data)

        if runall or cli_args['lr']:
            classifier = classifiers.log_reg_train(features, labels)
            classification = classifiers.log_reg_classify(
                classifier,
                validation_features)
            data.save_classification(
                classification,
                "./custom_lr_classification.csv",
                validation_mapping)

        if runall or cli_args['knn']:
            classifier = classifiers.knn_train(features, labels)
            classification = classifiers.knn_classify(
                classifier,
                validation_features)
            data.save_classification(
                classification,
                "./custom_knn_classification.csv",
                validation_mapping)

        if runall or cli_args['svm']:
            classifier = classifiers.svm_train(features, labels)
            classification = classifiers.svm_classify(
                classifier,
                validation_features)
            data.save_classification(
                classification,
                "./custom_svm_classification.csv",
                validation_mapping)
    return


if __name__ == "__main__":
    main()
