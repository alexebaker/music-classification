from __future__ import print_function
from __future__ import unicode_literals

from music_classifier import cli
from music_classifier import data
#from music_classifier import classifiers


def main():
    """Main entry point."""
    # Parse the command line arguments
    cli_args = cli.parse_args()

    audio_data = data.read_music_files()
    fft_features = data.get_fft_features(audio_data)
    #mfcc_features = data.get_mfcc_features(audio_data)
    return


if __name__ == "__main__":
    main()
