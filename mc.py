from __future__ import print_function
from __future__ import unicode_literals

from music_classifier import cli
from music_classifier import data
from music_classifier import classifiers


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
        features = data.get_fft_features(audio_data)
        validation_features = data.get_fft_features(validation_data)

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

    if runall or cli_args['mfcc']:
        features = data.get_mfcc_features(audio_data)
        validation_features = data.get_mfcc_features(validation_data)

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
    return


if __name__ == "__main__":
    main()
