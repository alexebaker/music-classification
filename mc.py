from __future__ import print_function
from __future__ import unicode_literals

from music_classifier import cli
from music_classifier import data
from music_classifier import classifiers


def main():
    """Main entry point."""
    # Parse the command line arguments
    cli_args = cli.parse_args()

    audio_data = data.read_music_files()
    file_data = audio_data[:, :-1]
    labels = audio_data[:, -1]
    fft_features = data.get_fft_features(file_data)
    #mfcc_features = data.get_mfcc_features(file_data)

    fft_lr = classifiers.log_reg_train(fft_features, labels)
    #mfcc_lr = classifiers.log_reg_train(mfcc_features, labels)
    #fft_knn = classifiers.knn_train(fft_features, labels)
    #mfcc_knn = classifiers.knn_train(mfcc_features, labels)

    validation_data, validation_mapping = data.read_validation_files()

    fft_validation_features = data.get_fft_features(validation_data)
    #mfcc_validation_features = data.get_mfcc_features(validation_data)

    fft_lr_classification = classifiers.log_reg_classify(fft_lr, fft_validation_features)
    #mfcc_lr_classification = classifiers.log_reg_classify(mfcc_lr, mfcc_validation_features)
    #fft_knn_classification = classifiers.knn_classify(fft_knn, fft_validation_features)
    #mfcc_knn_classification = classifiers.knn_classify(mfcc_knn, mfcc_validation_features)

    data.save_classification(fft_lr_classification, "./fft_lr_classification.csv", validation_mapping)
    #data.save_classification(mfcc_lr_classification, "./mfcc_lr_classification.csv", validation_mapping)
    #data.save_classification(fft_knn_classification, "./fft_knn_classification.csv", validation_mapping)
    #data.save_classification(mfcc_knn_classification, "./mfcc_knn_classification.csv", validation_mapping)
    return


if __name__ == "__main__":
    main()
