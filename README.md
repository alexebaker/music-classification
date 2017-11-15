# Music Classification

UNM CS 429/529 Machine Learning Project 3: Music Classification


## Getting started

This project uses several dependencies. You may need to use pip to install these dependencies.
It is recommended to use a virtualenv to run this code, however you can also use your system wide python as well.

If you do not want to use the virtualenv, install the requirements directly:

```bash
pip install -r requirements.txt
```

**or**

Use virtualenv. If virtualenv is not install on your system, install it with pip:

```bash
pip install virtualenv
```

Next, you want to create the virtual env in your current directory:

```bash
virtualenv .venv
```

Next, activate the virtualenv in your current shell:

```bash
source .venv/bin/activate
```

Now, install the python requirements:

```bash
pip install -r requirements.txt
```

You can deactivate the virtualenv with the following command, however, make sure the virtualenv is active when you run the code:

```bash
deactivate
```

## Details

Details about this project can be found on [Kaggle](https://inclass.kaggle.com/c/cs529-project3)


## Usage

**NOTE**: This code will work in python 2. It may work in python 3, but it has not been tested. Please use python 2 if python 3 does not work for you.

Make sure the virtualenv is active before you try running the python code. You can activate it by:

```bash
source .venv/bin/activate
```

Once the virtualenv is activated, you can run the python script. The main entry point for this project is `mc.py`. Use the `-h` flag from any command to see help:

```bash
>>>python mc.py -h
usage: mc.py [-h] [--training-data TRAINING_DATA]
             [--validation-data VALIDATION_DATA]
             [--features FEATURES [FEATURES ...]]
             [--methods METHODS [METHODS ...]] [--all] [--cross-validate]
             [--confusion-matrix]

Classifies the testing data using naive bayes and the training data.

optional arguments:
  -h, --help            show this help message and exit
  --training-data TRAINING_DATA
                        Path to the training data folder.
  --validation-data VALIDATION_DATA
                        Path to the validation data folder.
  --features FEATURES [FEATURES ...]
                        Feature extraction method to use. Can be multiple
                        features at once. Features can be: fft, mfcc, or dwt.
  --methods METHODS [METHODS ...]
                        Classification methods to use. Can be multiple methods
                        at once. Methods can be: lr, knn, svm, or nn.
  --all                 Run all features and classifiers.
  --cross-validate      Perform 10 fold cross validation.
  --confusion-matrix    Calculate the confusion matrix for the given methods.
```

You must specify at least one feature and one method to run. You can specify multiple if you wish:

```bash
python mc.py --features fft mfcc --methods svm nn
```

If you want to run all features and methods, you can use the `--all` flag. **NOTE** This will take a long time to run:

```bash
python mc.py --all
```

To perform cross validation, add the `--cross-validate` flag:

```bash
python mc.py --features fft mfcc --methods svm nn --cross-validate
```

To generate a confusion matrix, add the `--confusion-matrix` flag:

```bash
python mc.py --features fft mfcc --methods svm nn --confusion-matrix
```

if your music data files are in a different directory than `./data/`, you can specify them with the `--training-data` and `--validation-data` flags:


```bash
python mc.py --training-data /path/to/training/data --validation-data /path/to/validation/data --features fft mfcc --methods svm nn
```


## Documentation

This module uses documentation complied by [sphinx](http://www.sphinx-doc.org/en/stable/) located in the `docs/` directory. To build the documentation, run the Makefile:

```bash
source .venv/bin/activate
make docs
```

Once the documentation is built, it can be viewed in your brower by running the `open-docs.py` script:

```bash
python open-docs.py
```


## TODO

- [x] - Parse FFT features
- [x] - Parse MFCC features
- [x] - Parse custom features
- [x] - Classify FFT features
- [x] - Classify MFCC features
- [x] - Classify custom features
- [x] - Write up final report


## Authors

* [Alexander Baker](mailto:alexebaker@unm.edu)

* [Caleb Waters](mailto:waterscaleb@unm.edu)

* [Mark Mitchell](mailto:mamitchell@unm.edu)
